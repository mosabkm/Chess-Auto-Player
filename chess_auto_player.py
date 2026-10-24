#!/usr/bin/env python3
"""
Chess Auto-Player v4.3 - Fixed opponent detection stability
"""

import os
import sys
import time
import logging
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from collections import Counter

try:
    from PIL import ImageGrab, Image, ImageOps, ImageEnhance, ImageFilter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import ImageGrab, Image, ImageOps, ImageEnhance, ImageFilter

try:
    from chessimg2pos import predict_fen
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chessimg2pos"])
    from chessimg2pos import predict_fen

import numpy as np
import cv2
import chess
import chess.engine
import pyautogui

VERSION  = "4.3.0"
APP_NAME = "Chess Auto-Player"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


###############################################################################
# Screen Capture
###############################################################################

def capture_screen() -> Optional[np.ndarray]:
    """Capture full screen via PIL ImageGrab. Returns BGR numpy array."""
    try:
        pil_img = ImageGrab.grab()
        arr     = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"ImageGrab failed: {e}")
        return None


###############################################################################
# Board Detector
###############################################################################

class ChessBoardDetector:
    def __init__(self, debug: bool = False):
        self.debug     = debug
        self.debug_dir = "debug_images"
        self.manual_region: Optional[Tuple[int,int,int,int]] = None
        os.makedirs(self.debug_dir, exist_ok=True)

    def detect_board(self, image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        if self.manual_region:
            return self.manual_region
        return self._detect_by_grid(image) or self._detect_by_contour(image)

    def _detect_by_grid(self, image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        best_score, best_region = 0.0, None

        for size_frac in np.arange(0.2, 0.95, 0.05):
            board_size = (int(min(h, w) * size_frac) // 8) * 8
            if board_size < 80:
                continue
            sq   = board_size // 8
            step = max(sq // 2, 20)
            for y0 in range(0, h - board_size, step):
                for x0 in range(0, w - board_size, step):
                    score = self._grid_score(
                        gray[y0:y0+board_size, x0:x0+board_size], sq
                    )
                    if score > best_score:
                        best_score  = score
                        best_region = (x0, y0, board_size, board_size)

        if best_score > 0.55 and best_region:
            if self.debug:
                self._save_debug(image, best_region)
            return best_region
        return None

    def _grid_score(self, crop: np.ndarray, sq: int) -> float:
        h, w = crop.shape
        if h < sq * 8 or w < sq * 8:
            return 0.0
        mn, mx = float('inf'), float('-inf')
        means  = []
        for r in range(8):
            row = []
            for c in range(8):
                m = float(np.mean(crop[r*sq:(r+1)*sq, c*sq:(c+1)*sq]))
                row.append(m)
                if m < mn: mn = m
                if m > mx: mx = m
            means.append(row)
        if mx - mn < 20:
            return 0.0
        rng = mx - mn
        ok  = sum(
            int(((means[r][c] - mn) / rng > 0.5) == ((r + c) % 2 == 0))
            for r in range(8) for c in range(8)
        )
        return ok / 64.0

    def _detect_by_contour(self, image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        edges   = cv2.Canny(clahe.apply(gray), 30, 100)
        dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ih, iw  = image.shape[:2]
        best, best_s = None, 0.0
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(cnt) < 10_000:
                break
            x, y, bw, bh = cv2.boundingRect(cnt)
            if not (0.8 <= bw / bh <= 1.25):
                continue
            if bw < iw * 0.15 or bh < ih * 0.15:
                continue
            score = cv2.contourArea(cnt) / (bw * bh)
            if score > best_s:
                best_s, best = score, (x, y, bw, bh)
        return best if best and best_s > 0.5 else None

    def _save_debug(self, image: np.ndarray, region: Tuple):
        dbg = image.copy()
        x, y, w, h = region
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 3)
        sq = w // 8
        for i in range(1, 8):
            cv2.line(dbg, (x+i*sq, y), (x+i*sq, y+h), (0, 200, 0), 1)
            cv2.line(dbg, (x, y+i*sq), (x+w, y+i*sq), (0, 200, 0), 1)
        cv2.imwrite(f"{self.debug_dir}/board_detection.jpg", dbg)

    def crop_to_pil(self, image: np.ndarray,
                    region: Tuple[int,int,int,int]) -> Image.Image:
        x, y, w, h = region
        sq       = min(w, h) // 8 * 8
        crop_bgr = image[y:y+sq, x:x+sq]
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


###############################################################################
# FEN Reader
###############################################################################

class FENReader:
    """
    Wraps chessimg2pos.predict_fen() and uses an ensemble of preprocessing
    variants with square-by-square voting to reduce persistent piece misreads.
    """

    def __init__(self, debug_dir: str = "debug_images"):
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        self._tmp_path = os.path.join(debug_dir, "_board_tmp.jpg")

    @staticmethod
    def _fix_fen_placement(raw: str) -> str:
        fixed_rows = []
        for row in raw.split('/'):
            fixed_row = ''
            count = 0
            for ch in row:
                if ch == '1':
                    count += 1
                else:
                    if count:
                        fixed_row += str(count)
                        count = 0
                    fixed_row += ch
            if count:
                fixed_row += str(count)
            fixed_rows.append(fixed_row)
        return '/'.join(fixed_rows)

    @staticmethod
    def _validate_placement(placement: str) -> bool:
        rows = placement.split('/')
        if len(rows) != 8:
            return False
        for row in rows:
            count = 0
            for ch in row:
                if ch.isdigit():
                    count += int(ch)
                elif ch in 'pnbrqkPNBRQK':
                    count += 1
                else:
                    return False
            if count != 8:
                return False
        return True

    @staticmethod
    def _placement_to_grid(placement: str) -> Optional[List[List[str]]]:
        rows = placement.split('/')
        if len(rows) != 8:
            return None
        grid = []
        for row in rows:
            out = []
            for ch in row:
                if ch.isdigit():
                    out.extend(['.'] * int(ch))
                else:
                    out.append(ch)
            if len(out) != 8:
                return None
            grid.append(out)
        return grid

    @staticmethod
    def _grid_to_placement(grid: List[List[str]]) -> str:
        rows = []
        for row in grid:
            s = ''
            empty = 0
            for ch in row:
                if ch == '.':
                    empty += 1
                else:
                    if empty:
                        s += str(empty)
                        empty = 0
                    s += ch
            if empty:
                s += str(empty)
            rows.append(s)
        return '/'.join(rows)

    @staticmethod
    def _normalize_board_image(board_pil: Image.Image) -> Image.Image:
        w, h = board_pil.size
        s = min(w, h) // 8 * 8
        if s <= 0:
            return board_pil
        if (w, h) != (s, s):
            board_pil = board_pil.crop((0, 0, s, s))
        if s < 640:
            board_pil = board_pil.resize((640, 640), Image.Resampling.LANCZOS)
        return board_pil

    def _variants(self, board_pil: Image.Image):
        base = self._normalize_board_image(board_pil)
        gray = ImageOps.grayscale(base)
        autocontrast = ImageOps.autocontrast(base)
        high_contrast = ImageEnhance.Contrast(base).enhance(1.8)
        sharp = ImageEnhance.Sharpness(base).enhance(2.4)
        gray_contrast = ImageEnhance.Contrast(gray).enhance(2.0)
        gray_sharp = ImageEnhance.Sharpness(gray_contrast).enhance(2.5)
        denoise = base.filter(ImageFilter.MedianFilter(size=3))
        denoise_sharp = ImageEnhance.Sharpness(denoise).enhance(2.3)
        eq_gray = ImageOps.equalize(gray)
        return [
            ("base", base),
            ("autocontrast", autocontrast),
            ("high_contrast", high_contrast),
            ("sharp", sharp),
            ("gray", gray),
            ("gray_contrast", gray_contrast),
            ("gray_sharp", gray_sharp),
            ("denoise", denoise),
            ("denoise_sharp", denoise_sharp),
            ("eq_gray", eq_gray),
        ]

    def _predict_raw(self, img: Image.Image, tag: str) -> Optional[str]:
        try:
            img.save(self._tmp_path, "JPEG", quality=98)
            raw_fen = predict_fen(self._tmp_path)
            logger.debug(f"chessimg2pos raw [{tag}]: {raw_fen!r}")
            if not raw_fen:
                return None
            return raw_fen.strip().split()[0]
        except Exception as e:
            logger.debug(f"predict_fen failed for {tag}: {e}")
            return None

    def _collect_candidate_grids(self, board_pil: Image.Image):
        grids = []
        for tag, variant in self._variants(board_pil):
            raw_placement = self._predict_raw(variant, tag)
            if not raw_placement:
                continue
            fixed = self._fix_fen_placement(raw_placement)
            logger.debug(f"FEN fix [{tag}]: {raw_placement!r} -> {fixed!r}")
            if not self._validate_placement(fixed):
                continue
            grid = self._placement_to_grid(fixed)
            if grid is None:
                continue
            grids.append((tag, fixed, grid))
        return grids

    @staticmethod
    def _piece_vote_weight(piece: str, expected_piece: Optional[str]) -> float:
        w = 1.0
        if expected_piece is not None and piece == expected_piece:
            w += 2.0
        # Discourage king/queen flips a bit unless expected
        if piece in ('k', 'K', 'q', 'Q'):
            w += 0.2
        return w

    def _ensemble_grid(
        self,
        candidate_grids,
        expected_board: Optional[chess.Board] = None,
    ) -> Optional[List[List[str]]]:
        if not candidate_grids:
            return None

        expected_grid = None
        if expected_board is not None:
            expected_grid = self._placement_to_grid(expected_board.board_fen())

        out = [['.' for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                votes = Counter()
                expected_piece = expected_grid[r][c] if expected_grid else None
                for tag, fixed, grid in candidate_grids:
                    piece = grid[r][c]
                    votes[piece] += self._piece_vote_weight(piece, expected_piece)
                out[r][c] = votes.most_common(1)[0][0]
        return out

    @staticmethod
    def _legal_next_placements(
        expected_board: chess.Board,
        expected_turn: chess.Color
    ) -> set:
        tmp = expected_board.copy(stack=False)
        tmp.turn = expected_turn
        out = {expected_board.board_fen()}
        for mv in tmp.legal_moves:
            b = tmp.copy(stack=False)
            b.push(mv)
            out.add(b.board_fen())
        return out

    def _score_candidate(
        self,
        placement: str,
        expected_board: Optional[chess.Board],
        expected_turn: Optional[chess.Color],
    ) -> int:
        score = 0
        if expected_board is not None and expected_turn is not None:
            legal = self._legal_next_placements(expected_board, expected_turn)
            if placement == expected_board.board_fen():
                score += 12
            if placement in legal:
                score += 25

            exp_grid = self._placement_to_grid(expected_board.board_fen())
            cur_grid = self._placement_to_grid(placement)
            if exp_grid and cur_grid:
                same = sum(1 for r in range(8) for c in range(8) if exp_grid[r][c] == cur_grid[r][c])
                score += same

        if placement.count('k') == 1 and placement.count('K') == 1:
            score += 5
        if placement.count('q') <= 1 and placement.count('Q') <= 1:
            score += 2
        return score

    def board_image_to_fen(
        self,
        board_pil: Image.Image,
        player_turn: str = 'w',
        expected_board: Optional[chess.Board] = None,
        expected_turn: Optional[chess.Color] = None,
    ) -> Optional[str]:
        candidate_grids = self._collect_candidate_grids(board_pil)
        if not candidate_grids:
            logger.error("FENReader: no valid candidates from preprocessing variants")
            return None

        candidate_map = {}
        for tag, fixed, grid in candidate_grids:
            candidate_map.setdefault(fixed, []).append(tag)

        ensemble = self._ensemble_grid(candidate_grids, expected_board=expected_board)
        if ensemble is not None:
            ensemble_placement = self._grid_to_placement(ensemble)
            if self._validate_placement(ensemble_placement):
                candidate_map.setdefault(ensemble_placement, []).append('ensemble')

        scored = []
        for placement, tags in candidate_map.items():
            fen = f"{placement} {player_turn} - - 0 1"
            try:
                board = chess.Board(fen)
                board.turn = chess.WHITE if player_turn == 'w' else chess.BLACK
                score = self._score_candidate(placement, expected_board, expected_turn) + len(tags) * 3
                if 'ensemble' in tags:
                    score += 4
                scored.append((score, placement, tags, fen))
            except Exception:
                continue

        if not scored:
            logger.error("FENReader: all candidates rejected after scoring")
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_placement, best_tags, best_fen = scored[0]
        logger.info(f"Valid FEN {best_tags} score={best_score}: {best_fen}")
        logger.debug(f"Candidate summary: {[(s,p,t) for s,p,t,_ in scored[:5]]}")
        return best_fen

    def save_debug_board(self, board_pil: Image.Image, tag: str = "") -> str:
        ts = datetime.now().strftime('%H%M%S')
        path = f"{self.debug_dir}/board_{ts}{tag}.jpg"
        self._normalize_board_image(board_pil).save(path, "JPEG", quality=98)
        return path


###############################################################################
# FEN Validator
###############################################################################

class FENValidator:

    @staticmethod
    def is_safe(board: chess.Board) -> bool:
        try:
            if board.king(chess.WHITE) is None: return False
            if board.king(chess.BLACK) is None: return False
            for sq in chess.SQUARES:
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN:
                    if chess.square_rank(sq) in (0, 7):
                        return False
            opp      = board.copy()
            opp.turn = not board.turn
            return not opp.is_check()
        except Exception:
            return False

    @staticmethod
    def sanitize(board: chess.Board) -> Optional[chess.Board]:
        b = board.copy()
        for sq in list(chess.SQUARES):
            p = b.piece_at(sq)
            if (p and p.piece_type == chess.PAWN
                    and chess.square_rank(sq) in (0, 7)):
                b.remove_piece_at(sq)
        for color, default in (
            (chess.WHITE, chess.E1), (chess.BLACK, chess.E8)
        ):
            kings = [s for s in chess.SQUARES
                     if b.piece_at(s) == chess.Piece(chess.KING, color)]
            if not kings:
                b.set_piece_at(default, chess.Piece(chess.KING, color))
            elif len(kings) > 1:
                kings.sort(key=lambda s: abs(s - default))
                for extra in kings[1:]:
                    b.remove_piece_at(extra)
        b.castling_rights = chess.BB_EMPTY
        b.ep_square       = None
        return b if FENValidator.is_safe(b) else None


###############################################################################
# Auto-Player
###############################################################################

class ChessAutoPlayer:
    SETTLE_TIME      = 0.45   # wait after our move for animation
    POLL_INTERVAL    = 0.20   # seconds between FEN polls while waiting
    OPPONENT_TIMEOUT = 300.0  # 5 min max

    # Opponent detection tuning
    # How many reads to take per "vote window"
    VOTE_WINDOW      = 2
    # How many of those reads must agree on a placement for it to count
    VOTE_THRESHOLD   = 1
    # After this many seconds of any-change, accept best candidate
    CHANGE_ACCEPT_AFTER = 0.7
    # If a changed board appears immediately after our move, accept fast.
    FAST_REPLY_WINDOW = 1.5

    def __init__(self):
        self.detector   = ChessBoardDetector()
        self.fen_reader = FENReader()

        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.stockfish_path = self._find_stockfish()
        self.think_time     = 2.0
        self.engine_threads = max(1, (os.cpu_count() or 2) // 2)
        self.engine_hash    = 128
        self.engine_skill   = 20
        self.engine_elo     = 2500
        self.limit_strength = False
        self.ponder_enabled = False
        self.settle_time    = self.SETTLE_TIME
        self.poll_interval  = self.POLL_INTERVAL
        self.change_accept_after = self.CHANGE_ACCEPT_AFTER
        self.fast_reply_window   = self.FAST_REPLY_WINDOW
        self.player_color   = 'white'
        self.board_region:  Optional[Tuple[int,int,int,int]] = None
        self.is_running     = False
        self.move_counter   = 0
        self.move_history:  List[str] = []
        self.debug          = True
        self.last_fen:      Optional[str] = None

        self.status_callback = None
        self.move_callback   = None

    # ?? Stockfish ?????????????????????????????????????????????????????????????

    def _find_stockfish(self) -> str:
        for p in ('/usr/games/stockfish', '/usr/bin/stockfish',
                  '/usr/local/bin/stockfish', 'stockfish'):
            if os.path.isfile(p): return p
            try:
                r = subprocess.run(['which', p], capture_output=True, text=True)
                if r.returncode == 0: return r.stdout.strip()
            except Exception: pass
        return 'stockfish'

    def initialize_engine(self) -> bool:
        self._quit_engine()
        try:
            if not os.path.isfile(self.stockfish_path):
                self._status(f"? Stockfish not found: {self.stockfish_path}")
                return False
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.stockfish_path, timeout=10
            )
            cfg = {
                "Hash": int(self.engine_hash),
                "Threads": int(self.engine_threads),
            }
            try:
                cfg["Skill Level"] = int(self.engine_skill)
            except Exception:
                pass
            try:
                cfg["UCI_LimitStrength"] = bool(self.limit_strength)
                if self.limit_strength:
                    cfg["UCI_Elo"] = int(self.engine_elo)
            except Exception:
                pass
            self.engine.configure(cfg)
            self.engine.ping()
            self._status(
                f"? Stockfish ready | threads={self.engine_threads} hash={self.engine_hash} "
                f"skill={self.engine_skill} limit={'on' if self.limit_strength else 'off'} "
                f"elo={self.engine_elo if self.limit_strength else 'max'}"
            )
            return True
        except Exception as e:
            logger.error(f"Engine init: {e}")
            return False

    def _quit_engine(self):
        if self.engine:
            try: self.engine.quit()
            except Exception: pass
            self.engine = None

    def get_ai_move(self, board: chess.Board) -> Optional[chess.Move]:
        safe = FENValidator.sanitize(board)
        if safe is None:
            self._status("? Position unsafe for engine")
            return None
        if safe.is_game_over() or not list(safe.legal_moves):
            return None
        if not self.engine:
            if not self.initialize_engine(): return None
        try:
            self.engine.ping()
            result = self.engine.play(
                safe, chess.engine.Limit(time=self.think_time)
            )
            return result.move
        except chess.engine.EngineTerminatedError as e:
            logger.error(f"Engine crashed: {e}")
            self._quit_engine(); time.sleep(0.5); self.initialize_engine()
            return None
        except Exception as e:
            logger.error(f"Engine error: {e}")
            self._quit_engine(); return None

    # ?? board reading ?????????????????????????????????????????????????????????

    def read_board_from_screen(
        self,
        turn: Optional[chess.Color] = None,
        expected_board: Optional[chess.Board] = None,
        expected_turn: Optional[chess.Color] = None,
    ) -> Optional[chess.Board]:
        """
        Capture ? crop ? chessimg2pos (with FEN fix) ? chess.Board.
        turn: explicitly sets which side is to move (always pass this).
        """
        screen = capture_screen()
        if screen is None:
            return None

        region = self.board_region or self.detector.detect_board(screen)
        if region is None:
            return None
        self.board_region = region

        board_pil  = self.detector.crop_to_pil(screen, region)
        turn_color = (turn if turn is not None
                      else (chess.WHITE
                            if len(self.move_history) % 2 == 0
                            else chess.BLACK))
        turn_char  = 'w' if turn_color == chess.WHITE else 'b'

        fen = self.fen_reader.board_image_to_fen(board_pil, turn_char, expected_board=expected_board, expected_turn=expected_turn)
        if fen is None:
            return None

        try:
            board      = chess.Board(fen)
            board.turn = turn_color
            self.last_fen = board.fen()
            return board
        except Exception as e:
            logger.error(f"chess.Board() failed: {e} | FEN={fen}")
            return None

    def _read_placement(
        self,
        turn: chess.Color,
        expected_board: Optional[chess.Board] = None,
        expected_turn: Optional[chess.Color] = None,
    ) -> Optional[str]:
        """Read board and return only the piece-placement string, or None."""
        board = self.read_board_from_screen(
            turn=turn,
            expected_board=expected_board,
            expected_turn=expected_turn,
        )
        if board is None:
            return None
        return board.board_fen()

    def get_board_pil(self) -> Optional[Image.Image]:
        screen = capture_screen()
        if screen is None: return None
        region = self.board_region or self.detector.detect_board(screen)
        if region is None: return None
        self.board_region = region
        return self.detector.crop_to_pil(screen, region)

    # ?? mouse ?????????????????????????????????????????????????????????????????

    def square_to_screen(self, square_name: str) -> Tuple[int, int]:
        if not self.board_region:
            raise ValueError("Board region not set")
        x, y, w, h = self.board_region
        sq_size  = min(w, h) // 8
        file_idx = ord(square_name[0]) - ord('a')
        rank_idx = 8 - int(square_name[1])
        if self.player_color == 'black':
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx
        return (x + file_idx * sq_size + sq_size // 2,
                y + rank_idx * sq_size + sq_size // 2)

    def execute_move(self, move: chess.Move) -> bool:
        try:
            from_n = chess.square_name(move.from_square)
            to_n   = chess.square_name(move.to_square)
            fx, fy = self.square_to_screen(from_n)
            tx, ty = self.square_to_screen(to_n)
            logger.info(f"Mouse: {from_n}({fx},{fy}) ? {to_n}({tx},{ty})")
            pyautogui.moveTo(fx, fy, duration=0.25); time.sleep(0.08)
            pyautogui.mouseDown();                   time.sleep(0.12)
            pyautogui.moveTo(tx, ty, duration=0.35); time.sleep(0.12)
            pyautogui.mouseUp();                     time.sleep(0.15)
            if move.promotion:
                time.sleep(0.4)
                pyautogui.press(
                    {chess.QUEEN: 'q', chess.ROOK: 'r',
                     chess.BISHOP: 'b', chess.KNIGHT: 'n'
                     }.get(move.promotion, 'q')
                )
            if self.move_callback:
                self.move_callback(from_n, to_n, move.promotion)
            return True
        except Exception as e:
            logger.error(f"execute_move: {e}"); return False

    # ?? confirm our move appeared on screen ???????????????????????????????????

    def _confirm_move(
        self,
        move: chess.Move,
        turn_after: chess.Color,
        retries: int = 10
    ) -> Optional[chess.Board]:
        """
        Confirm OUR move only, without accidentally absorbing the opponent reply.

        We compare fresh reads against the locally expected board after pushing our
        move. A read is accepted only if it exactly matches that expected position.
        If the opponent has already moved too fast, we return the local expected
        board and let _wait_for_opponent() detect the reply separately.
        """
        expected_before = self.read_board_from_screen(turn=not turn_after)
        if expected_before is None:
            expected_before = None

        expected_after = None
        if expected_before is not None:
            try:
                expected_after = expected_before.copy(stack=False)
                expected_after.push(move)
                expected_after.turn = turn_after
            except Exception:
                expected_after = None

        # Fallback local expectation from last known state if available later.
        readings: List[str] = []
        saw_opponent_reply = False

        for attempt in range(retries):
            time.sleep(0.45)
            board = self.read_board_from_screen(
                turn=turn_after,
                expected_board=expected_after,
                expected_turn=turn_after,
            )
            if board is None:
                continue

            placement = board.board_fen()
            readings.append(placement)

            # Exact match to our expected post-move board => confirmed.
            if expected_after is not None and placement == expected_after.board_fen():
                board.turn = turn_after
                self.last_fen = board.fen()
                logger.info(f"Move confirmed on attempt {attempt+1}")
                return board

            # If our from-square is empty but board differs, the opponent may have
            # already moved; do not absorb that here.
            if board.piece_at(move.from_square) is None:
                saw_opponent_reply = True
                logger.info(
                    f"Confirm saw moved-away piece but board differs on attempt {attempt+1}; "
                    f"deferring to opponent detection"
                )
                break

            logger.debug(
                f"Confirm attempt {attempt+1}: board does not yet match expected post-move state"
            )

        if expected_after is not None:
            logger.warning("Move confirm fallback: using local expected post-move board")
            self.last_fen = expected_after.fen()
            return expected_after

        # Last resort: do not accept a random changed board here.
        if saw_opponent_reply and readings:
            logger.warning("Move confirm fallback: opponent may have moved too quickly; using last read conservatively")
        return None

    # ?? wait for opponent ?????????????????????????????????????????????????????

    def _wait_for_opponent(
        self,
        current_board: chess.Board,
        stop_event: Optional[threading.Event],
        ai_color: chess.Color
    ) -> Optional[chess.Board]:
        """
        Poll until the board position changes from old_placement.

        Strategy (fixes the "stuck waiting" bug):
        ??????????????????????????????????????????
        We collect readings into a rolling window.
        At each step we ask: "does the MAJORITY of recent readings show
        a placement DIFFERENT from old_placement?"

        If yes for VOTE_THRESHOLD out of VOTE_WINDOW reads ? confirmed.

        Additionally: if ANY different placement has been seen for more
        than CHANGE_ACCEPT_AFTER seconds, we accept the most-common
        non-old placement seen so far (handles noisy OCR).
        """
        wait_start    = time.time()
        last_log_time = time.time()
        old_placement = current_board.board_fen()

        logger.info(f"Waiting for opponent. Old: {old_placement}")

        # Rolling window of recent placements
        window:         List[str]          = []
        # All non-old placements seen, with first-seen timestamp
        candidates:     Dict[str, float]   = {}
        # Best candidate so far
        best_candidate: Optional[str]      = None

        while self.is_running:
            if stop_event and stop_event.is_set():
                return None

            elapsed = time.time() - wait_start

            if elapsed > self.OPPONENT_TIMEOUT:
                self._status(f"? Opponent timeout ({self.OPPONENT_TIMEOUT:.0f}s)")
                return None

            if time.time() - last_log_time > 15:
                self._status(f"Still waiting ? ({elapsed:.0f}s)")
                last_log_time = time.time()

            # ?? take one reading ??????????????????????????????????????????????
            placement = self._read_placement(
                turn=ai_color,
                expected_board=current_board,
                expected_turn=not ai_color,
            )

            if placement is not None:
                window.append(placement)
                if len(window) > self.VOTE_WINDOW:
                    window.pop(0)

                # ?? count how many window reads differ from old ????????????????
                diff_reads = [p for p in window if p != old_placement]
                same_reads = [p for p in window if p == old_placement]

                logger.debug(
                    f"Window: {len(diff_reads)} different, "
                    f"{len(same_reads)} same | elapsed={elapsed:.1f}s"
                )

                if diff_reads:
                    # Track each candidate and when it first appeared
                    for p in diff_reads:
                        if p not in candidates:
                            candidates[p] = time.time()

                    # Most common non-old placement in window
                    best_candidate = Counter(diff_reads).most_common(1)[0][0]
                    best_count     = Counter(diff_reads).most_common(1)[0][1]

                    logger.debug(
                        f"Best candidate ({best_count}/{len(window)} reads): "
                        f"{best_candidate}"
                    )

                    # ?? Accept if: majority vote reached ??????????????????????
                    if best_count >= self.VOTE_THRESHOLD:
                        return self._accept_candidate(
                            best_candidate, ai_color, elapsed, "majority vote"
                        )

                    # ?? Accept if: candidate has been around long enough ???????
                    oldest = min(candidates.values())
                    age    = time.time() - oldest
                    if age >= self.change_accept_after and best_candidate:
                        return self._accept_candidate(
                            best_candidate, ai_color, elapsed,
                            f"timeout fallback ({age:.1f}s)"
                        )
                else:
                    # All window reads match old ? reset candidates
                    if candidates:
                        logger.debug("All window reads match old ? resetting candidates")
                    candidates.clear()
                    best_candidate = None

            time.sleep(self.poll_interval)

        return None

    def _accept_candidate(
        self,
        placement: str,
        ai_color: chess.Color,
        elapsed: float,
        reason: str
    ) -> Optional[chess.Board]:
        """Build a chess.Board from an accepted placement and set turn."""
        try:
            turn_char = 'w' if ai_color == chess.WHITE else 'b'
            fen       = f"{placement} {turn_char} - - 0 1"
            board     = chess.Board(fen)
            board.turn = ai_color
            self.last_fen = board.fen()

            logger.info(
                f"Opponent move accepted after {elapsed:.1f}s [{reason}]\n"
                f"  NEW: {placement}\n"
                f"  NEXT TURN: {'white' if ai_color == chess.WHITE else 'black'}"
            )

            if self.debug:
                pil = self.get_board_pil()
                if pil:
                    self.fen_reader.save_debug_board(pil, "_opponent")

            return board
        except Exception as e:
            logger.error(f"_accept_candidate failed: {e}")
            return None

    # ?? main loop ?????????????????????????????????????????????????????????????

    def play_auto(
        self,
        color: str = 'white',
        stop_event: Optional[threading.Event] = None
    ):
        self.player_color = color
        self.is_running   = True
        self.move_counter = 0
        self.move_history = []
        self.last_fen     = None
        self.board_region = None

        ai_color       = chess.WHITE if color == 'white' else chess.BLACK
        opponent_color = not ai_color

        self._status(f"Starting as {color} ?")
        if not self.initialize_engine():
            self.is_running = False
            return

        # ?? read initial board ????????????????????????????????????????????????
        self._status("Reading initial board position ?")
        current_board = None
        for attempt in range(20):
            if stop_event and stop_event.is_set():
                self.is_running = False; return
            current_board = self.read_board_from_screen(turn=ai_color)
            if current_board is not None:
                break
            self._status(f"Board read failed (attempt {attempt+1}/20) ?")
            time.sleep(1.5)

        if current_board is None:
            self._status("? Cannot read board ? check region / chessimg2pos")
            self.is_running = False; self._quit_engine(); return

        self._status(f"? Board read OK: {self.last_fen[:50]} ?")

        # ?? game loop ?????????????????????????????????????????????????????????
        while self.is_running:
            if stop_event and stop_event.is_set():
                break

            if current_board.is_game_over():
                self._status(f"Game over: {current_board.result()}")
                break

            if current_board.turn == ai_color:
                # ?? OUR TURN ??????????????????????????????????????????????????
                self.move_counter += 1
                self._status(f"Move {self.move_counter}: thinking ?")

                move = self.get_ai_move(current_board)
                if move is None:
                    self._status("? No move ? re-reading in 2s")
                    time.sleep(2.0)
                    b = self.read_board_from_screen(turn=ai_color)
                    if b: current_board = b
                    continue

                try:    san = current_board.san(move)
                except: san = move.uci()

                self._status(f"Playing: {san} ({move.uci()})")

                if self.execute_move(move):
                    self.move_history.append(move.uci())
                    self._status(f"? {san} played ? settling ?")
                    time.sleep(self.settle_time)

                    confirmed = self._confirm_move(
                        move,
                        turn_after=opponent_color,
                        retries=10
                    )

                    if confirmed:
                        current_board = confirmed
                        self._status(
                            f"? Board confirmed ? waiting for opponent ?"
                        )
                    else:
                        self._status("? Board confirm unclear ? advancing locally")
                        current_board.push(move)
                        current_board.turn = opponent_color
                        self.last_fen = current_board.fen()

                else:
                    self._status("? Mouse move failed")
                    time.sleep(1.0)

            else:
                # ?? OPPONENT'S TURN ???????????????????????????????????????????
                self._status("Waiting for opponent ?")
                new_board = self._wait_for_opponent(
                    current_board,
                    stop_event,
                    ai_color=ai_color
                )
                if new_board is not None:
                    # Safety: always force turn to ours
                    new_board.turn = ai_color
                    current_board  = new_board
                    self._status("Opponent moved ? our turn ?")
                else:
                    break

            time.sleep(0.1)

        self._quit_engine()
        self._status(f"Stopped after {self.move_counter} move(s).")
        self.is_running = False

    def _status(self, msg: str):
        logger.info(msg)
        if self.status_callback:
            self.status_callback(msg)


###############################################################################
# GUI
###############################################################################

class ChessAutoPlayerGUI:
    TAG_COLORS = {
        'info':    'black',
        'success': '#1a7a1a',
        'warning': '#b36b00',
        'error':   '#cc0000',
        'move':    '#0055cc',
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        ww = min(980, sw - 40); wh = min(680, sh - 80)
        root.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
        root.minsize(700, 500)
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.player = ChessAutoPlayer()

        self.player_color = tk.StringVar(value="white")
        self.think_time   = tk.DoubleVar(value=2.0)
        self.engine_threads = tk.IntVar(value=self.player.engine_threads)
        self.engine_hash = tk.IntVar(value=self.player.engine_hash)
        self.engine_skill = tk.IntVar(value=self.player.engine_skill)
        self.engine_elo = tk.IntVar(value=self.player.engine_elo)
        self.limit_strength = tk.BooleanVar(value=self.player.limit_strength)
        self.ponder_enabled = tk.BooleanVar(value=self.player.ponder_enabled)
        self.settle_time  = tk.DoubleVar(value=self.player.SETTLE_TIME)
        self.poll_interval = tk.DoubleVar(value=self.player.POLL_INTERVAL)
        self.fast_reply_window = tk.DoubleVar(value=self.player.FAST_REPLY_WINDOW)
        self.debug_mode   = tk.BooleanVar(value=True)

        self.status_queue = queue.Queue()
        self.stop_event   = threading.Event()
        self.auto_thread: Optional[threading.Thread] = None
        self.is_playing   = False

        self.player.debug           = True
        self.player.status_callback = self._enqueue_status
        self.player.move_callback   = self._on_move

        self._setup_styles()
        self._build_ui()
        self._poll_status()
        self._check_stockfish()
        self._check_chessimg2pos()

        self.log(f"? {APP_NAME} v{VERSION} ready", 'success')
        self.log("Fix: majority-vote + timeout-fallback opponent detection", 'info')

    def _setup_styles(self):
        s = ttk.Style(); s.theme_use('clam')
        s.configure('TLabelframe.Label', font=('Helvetica', 9, 'bold'))
        s.configure('Start.TButton', font=('Helvetica', 10, 'bold'),
                    foreground='white', background='#27ae60')
        s.configure('Stop.TButton', font=('Helvetica', 10, 'bold'),
                    foreground='white', background='#c0392b')
        s.configure('Sm.TButton', font=('Helvetica', 9))

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg='#2c3e50', pady=5)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"? {APP_NAME} v{VERSION}",
                 font=('Helvetica', 12, 'bold'),
                 bg='#2c3e50', fg='white').pack(side=tk.LEFT, padx=10)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(hdr, textvariable=self.status_var, font=('Helvetica', 9),
                 bg='#2c3e50', fg='#aed6f1', anchor='w').pack(
                     side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        nb_wrap = tk.Frame(main, width=300)
        nb_wrap.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        nb_wrap.pack_propagate(False)

        self.nb = ttk.Notebook(nb_wrap)
        self.nb.pack(fill=tk.BOTH, expand=True)

        t1 = tk.Frame(self.nb, padx=5, pady=5)
        t2 = tk.Frame(self.nb, padx=5, pady=5)
        t3 = tk.Frame(self.nb, padx=5, pady=5)
        self.nb.add(t1, text="? Play")
        self.nb.add(t2, text="? Board")
        self.nb.add(t3, text="? History")

        self._build_play_tab(t1)
        self._build_board_tab(t2)
        self._build_history_tab(t3)
        self._build_log(main)

    def _build_play_tab(self, p: tk.Frame):
        sf = ttk.LabelFrame(p, text="Engine", padding=5)
        sf.pack(fill=tk.X, pady=(0, 5))
        self.sf_var = tk.StringVar(value=self.player.stockfish_path)
        ttk.Entry(sf, textvariable=self.sf_var,
                  font=('Courier', 8)).pack(fill=tk.X, pady=(0, 3))
        row = tk.Frame(sf); row.pack(fill=tk.X)
        ttk.Button(row, text="Browse", style='Sm.TButton',
                   command=self._browse_sf).pack(
                       side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(row, text="Verify", style='Sm.TButton',
                   command=self._check_stockfish).pack(
                       side=tk.LEFT, expand=True, fill=tk.X)
        self.sf_lbl = ttk.Label(sf, text="", font=('Helvetica', 8))
        self.sf_lbl.pack(anchor='w', pady=(2, 0))

        col = ttk.LabelFrame(p, text="Play As", padding=5)
        col.pack(fill=tk.X, pady=(0, 5))
        row2 = tk.Frame(col); row2.pack()
        ttk.Radiobutton(row2, text="? White",
                        variable=self.player_color,
                        value="white").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(row2, text="? Black",
                        variable=self.player_color,
                        value="black").pack(side=tk.LEFT, padx=8)

        eng = ttk.LabelFrame(p, text="Think Time", padding=5)
        eng.pack(fill=tk.X, pady=(0, 5))
        row3 = tk.Frame(eng); row3.pack(fill=tk.X)
        ttk.Scale(row3, from_=0.5, to=15.0, variable=self.think_time,
                  orient=tk.HORIZONTAL).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.think_lbl = tk.Label(row3, text="2.0s", width=5,
                                  font=('Helvetica', 8))
        self.think_lbl.pack(side=tk.LEFT)
        self.think_time.trace_add('write', self._upd_think)

        eng2 = ttk.LabelFrame(p, text="Stockfish Options", padding=5)
        eng2.pack(fill=tk.X, pady=(0, 5))

        er1 = tk.Frame(eng2); er1.pack(fill=tk.X)
        tk.Label(er1, text="Threads", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(er1, from_=1, to=max(1, os.cpu_count() or 4), variable=self.engine_threads,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.threads_lbl = tk.Label(er1, text=str(self.player.engine_threads), width=5, font=('Helvetica', 8))
        self.threads_lbl.pack(side=tk.LEFT)
        self.engine_threads.trace_add('write', self._upd_engine)

        er2 = tk.Frame(eng2); er2.pack(fill=tk.X, pady=(3,0))
        tk.Label(er2, text="Hash", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(er2, from_=16, to=1024, variable=self.engine_hash,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.hash_lbl = tk.Label(er2, text=str(self.player.engine_hash), width=5, font=('Helvetica', 8))
        self.hash_lbl.pack(side=tk.LEFT)
        self.engine_hash.trace_add('write', self._upd_engine)

        er3 = tk.Frame(eng2); er3.pack(fill=tk.X, pady=(3,0))
        tk.Label(er3, text="Skill", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(er3, from_=0, to=20, variable=self.engine_skill,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.skill_lbl = tk.Label(er3, text=str(self.player.engine_skill), width=5, font=('Helvetica', 8))
        self.skill_lbl.pack(side=tk.LEFT)
        self.engine_skill.trace_add('write', self._upd_engine)

        er4 = tk.Frame(eng2); er4.pack(fill=tk.X, pady=(3,0))
        tk.Label(er4, text="Elo", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(er4, from_=1350, to=2850, variable=self.engine_elo,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.elo_lbl = tk.Label(er4, text=str(self.player.engine_elo), width=5, font=('Helvetica', 8))
        self.elo_lbl.pack(side=tk.LEFT)
        self.engine_elo.trace_add('write', self._upd_engine)

        er5 = tk.Frame(eng2); er5.pack(fill=tk.X, pady=(4,0))
        ttk.Checkbutton(er5, text="Limit strength (use Elo)", variable=self.limit_strength,
                        command=self._upd_engine).pack(anchor='w')
        ttk.Checkbutton(er5, text="Enable pondering", variable=self.ponder_enabled,
                        command=self._upd_engine).pack(anchor='w')
        ttk.Button(er5, text="Reinitialize Engine", style='Sm.TButton',
                   command=self._reinit_engine).pack(fill=tk.X, pady=(4,0))

        speed = ttk.LabelFrame(p, text="Speed / Detection", padding=5)
        speed.pack(fill=tk.X, pady=(0, 5))

        r1 = tk.Frame(speed); r1.pack(fill=tk.X)
        tk.Label(r1, text="Settle", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(r1, from_=0.1, to=2.0, variable=self.settle_time,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.settle_lbl = tk.Label(r1, text=f"{self.player.SETTLE_TIME:.2f}s", width=6, font=('Helvetica', 8))
        self.settle_lbl.pack(side=tk.LEFT)
        self.settle_time.trace_add('write', self._upd_speed)

        r2 = tk.Frame(speed); r2.pack(fill=tk.X, pady=(3,0))
        tk.Label(r2, text="Poll", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(r2, from_=0.05, to=1.0, variable=self.poll_interval,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.poll_lbl = tk.Label(r2, text=f"{self.player.POLL_INTERVAL:.2f}s", width=6, font=('Helvetica', 8))
        self.poll_lbl.pack(side=tk.LEFT)
        self.poll_interval.trace_add('write', self._upd_speed)

        r3 = tk.Frame(speed); r3.pack(fill=tk.X, pady=(3,0))
        tk.Label(r3, text="Fast reply", width=10, anchor='w').pack(side=tk.LEFT)
        ttk.Scale(r3, from_=0.2, to=3.0, variable=self.fast_reply_window,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fast_lbl = tk.Label(r3, text=f"{self.player.FAST_REPLY_WINDOW:.2f}s", width=6, font=('Helvetica', 8))
        self.fast_lbl.pack(side=tk.LEFT)
        self.fast_reply_window.trace_add('write', self._upd_speed)

        ctrl = ttk.LabelFrame(p, text="Control", padding=5)
        ctrl.pack(fill=tk.X, pady=(0, 5))
        self.start_btn = ttk.Button(ctrl, text="?  Start Auto-Play",
                                    command=self._start, style='Start.TButton')
        self.start_btn.pack(fill=tk.X, pady=(0, 3))
        self.stop_btn  = ttk.Button(ctrl, text="?  Stop",
                                    command=self._stop, style='Stop.TButton',
                                    state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)
        self.moves_lbl = tk.Label(ctrl, text="Moves: 0",
                                  font=('Helvetica', 10, 'bold'))
        self.moves_lbl.pack(pady=(4, 0))

        fen_f = ttk.LabelFrame(p, text="Current FEN", padding=5)
        fen_f.pack(fill=tk.X, pady=(0, 5))
        self.fen_var = tk.StringVar(value="?")
        tk.Label(fen_f, textvariable=self.fen_var, font=('Courier', 7),
                 wraplength=270, justify=tk.LEFT,
                 foreground='#2c3e50').pack(anchor='w')
        ttk.Button(fen_f, text="? Read FEN Now", style='Sm.TButton',
                   command=self._read_fen_now).pack(fill=tk.X, pady=(4, 0))

    def _build_board_tab(self, p: tk.Frame):
        lib = ttk.LabelFrame(p, text="chessimg2pos Library", padding=5)
        lib.pack(fill=tk.X, pady=(0, 5))
        self.lib_lbl = tk.Label(lib, text="Checking ?",
                                font=('Helvetica', 8), justify=tk.LEFT)
        self.lib_lbl.pack(anchor='w')
        ttk.Button(lib, text="? Re-check", style='Sm.TButton',
                   command=self._check_chessimg2pos).pack(
                       fill=tk.X, pady=(3, 0))

        det = ttk.LabelFrame(p, text="Board Region", padding=5)
        det.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(det, text="? Auto-Detect", style='Sm.TButton',
                   command=self._detect_board).pack(fill=tk.X, pady=(0, 3))
        self.board_lbl = tk.Label(det, text="No region set",
                                  foreground='gray', font=('Helvetica', 8),
                                  wraplength=270, justify=tk.LEFT)
        self.board_lbl.pack(anchor='w')

        man = ttk.LabelFrame(p, text="Manual Region (pixels)", padding=5)
        man.pack(fill=tk.X, pady=(0, 5))
        self._cal: Dict[str, tk.StringVar] = {}
        defaults = {'x': '180', 'y': '260', 'w': '304', 'h': '304'}
        grid_f = tk.Frame(man); grid_f.pack(fill=tk.X)
        for i, (lbl, key) in enumerate(
            (("X:", 'x'), ("Y:", 'y'), ("W:", 'w'), ("H:", 'h'))
        ):
            tk.Label(grid_f, text=lbl, font=('Helvetica', 9)).grid(
                row=i // 2, column=(i % 2) * 2,
                sticky='e', padx=(0, 2), pady=1
            )
            v = tk.StringVar(value=defaults[key])
            ttk.Entry(grid_f, textvariable=v, width=7,
                      font=('Courier', 9)).grid(
                          row=i // 2, column=(i % 2) * 2 + 1,
                          sticky='w', pady=1
                      )
            self._cal[key] = v
        ttk.Button(man, text="? Apply", style='Sm.TButton',
                   command=self._apply_manual).pack(fill=tk.X, pady=(5, 0))

        dbg = ttk.LabelFrame(p, text="Debug", padding=5)
        dbg.pack(fill=tk.X, pady=(0, 5))
        ttk.Checkbutton(dbg, text="Save debug images",
                        variable=self.debug_mode,
                        command=self._toggle_debug).pack(anchor='w')
        ttk.Button(dbg, text="? Capture Board Now", style='Sm.TButton',
                   command=self._save_debug).pack(fill=tk.X, pady=(3, 0))

    def _build_history_tab(self, p: tk.Frame):
        f = tk.Frame(p); f.pack(fill=tk.BOTH, expand=True)
        self.hist_box = tk.Listbox(f, font=('Courier', 9))
        sb = ttk.Scrollbar(f, orient=tk.VERTICAL, command=self.hist_box.yview)
        self.hist_box.configure(yscrollcommand=sb.set)
        self.hist_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(p, text="Clear", style='Sm.TButton',
                   command=lambda: self.hist_box.delete(0, tk.END)
                   ).pack(fill=tk.X, pady=(4, 0))

    def _build_log(self, parent: tk.Frame):
        lf = ttk.LabelFrame(parent, text="Activity Log", padding=4)
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_box = scrolledtext.ScrolledText(
            lf, font=('Courier', 8), state=tk.DISABLED, wrap=tk.WORD
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        for tag, fg in self.TAG_COLORS.items():
            self.log_box.tag_configure(tag, foreground=fg)
        row = tk.Frame(lf); row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(row, text="Clear", style='Sm.TButton',
                   command=self._clear_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Save", style='Sm.TButton',
                   command=self._save_log).pack(side=tk.LEFT, padx=2)

    def log(self, msg: str, level: str = 'info'):
        self.log_box.configure(state=tk.NORMAL)
        ts  = datetime.now().strftime('%H:%M:%S')
        tag = level if level in self.TAG_COLORS else 'info'
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _clear_log(self):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.delete('1.0', tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Text', '*.txt'), ('All', '*.*')]
        )
        if path:
            with open(path, 'w') as f:
                f.write(self.log_box.get('1.0', tk.END))
            self.log(f"Saved ? {path}", 'success')

    def _enqueue_status(self, msg: str):
        self.status_queue.put(msg)

    def _poll_status(self):
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self.status_var.set(msg[:90])
                lvl = ('error'   if any(c in msg for c in ('?', 'crash', 'failed'))
                  else 'success' if any(c in msg for c in ('?', '?', 'wins'))
                  else 'warning' if '?' in msg
                  else 'move'    if 'Playing' in msg
                  else 'info')
                self.log(msg, lvl)
                self.moves_lbl.config(text=f"Moves: {self.player.move_counter}")
                if self.player.last_fen:
                    self.fen_var.set(self.player.last_fen[:60] + " ?")
        except queue.Empty:
            pass
        self.root.after(150, self._poll_status)

    def _on_move(self, from_sq: str, to_sq: str, promo):
        n   = len(self.player.move_history)
        txt = f"{n:>3}. {from_sq}?{to_sq}"
        if promo:
            txt += '=' + {
                chess.QUEEN: 'Q', chess.ROOK: 'R',
                chess.BISHOP: 'B', chess.KNIGHT: 'N'
            }.get(promo, '?')
        self.hist_box.insert(tk.END, txt)
        self.hist_box.see(tk.END)

    def _browse_sf(self):
        p = filedialog.askopenfilename(title="Select Stockfish")
        if p:
            self.sf_var.set(p)
            self.player.stockfish_path = p
            self._check_stockfish()

    def _check_stockfish(self):
        path = self.sf_var.get()
        self.player.stockfish_path = path
        found = os.path.isfile(path)
        self.sf_lbl.config(
            text="? Found" if found else "? Not found",
            foreground='green' if found else 'red'
        )
        self.log(
            f"Stockfish {'found' if found else 'NOT found'}: {path}",
            'success' if found else 'error'
        )

    def _check_chessimg2pos(self):
        try:
            from chessimg2pos import predict_fen  # noqa: F401
            self.lib_lbl.config(
                text="? chessimg2pos installed\n"
                     "FEN fix active: 1-per-sq ? run-length",
                foreground='green'
            )
            self.log("? chessimg2pos available", 'success')
        except ImportError:
            self.lib_lbl.config(
                text="? Not installed\npip install chessimg2pos",
                foreground='red'
            )
            self.log("? chessimg2pos missing ? pip install chessimg2pos",
                     'error')

    def _detect_board(self):
        self.log("Auto-detecting board ?", 'info')
        def _run():
            img    = capture_screen()
            region = (self.player.detector.detect_board(img)
                      if img is not None else None)
            self.root.after(0, lambda: self._show_detection(region))
        threading.Thread(target=_run, daemon=True).start()

    def _show_detection(self, region):
        if region:
            x, y, w, h = region
            txt = f"({x},{y})  {w}×{h}  sq?{w//8}px"
            self.board_lbl.config(text=f"? {txt}", foreground='green')
            self.player.board_region = region
            for k, v in zip(('x', 'y', 'w', 'h'), region):
                self._cal[k].set(str(v))
            self.log(f"? Board: {txt}", 'success')
        else:
            self.board_lbl.config(text="? Not detected ? use manual",
                                  foreground='red')
            self.log("? Auto-detect failed", 'error')

    def _apply_manual(self):
        try:
            region = tuple(
                int(self._cal[k].get()) for k in ('x', 'y', 'w', 'h')
            )
            self.player.board_region = region
            self.player.detector.manual_region = region
            x, y, w, h = region
            self.board_lbl.config(
                text=f"Manual: ({x},{y}) {w}×{h}  sq?{w//8}px",
                foreground='blue'
            )
            self.log(f"Manual region: {region}", 'info')
        except ValueError:
            messagebox.showerror("Error", "All values must be integers")

    def _read_fen_now(self):
        self.log("Reading FEN ?", 'info')
        def _run():
            board = self.player.read_board_from_screen(turn=chess.WHITE)
            if board:
                fen = board.fen()
                self.root.after(0, lambda: (
                    self.log(f"FEN: {fen}", 'success'),
                    self.fen_var.set(fen[:60] + " ?")
                ))
            else:
                self.root.after(0, lambda:
                    self.log("? FEN read failed", 'error'))
        threading.Thread(target=_run, daemon=True).start()

    def _save_debug(self):
        def _run():
            img    = capture_screen()
            region = self.player.board_region
            if img is not None and region:
                dbg = img.copy()
                x, y, w, h = region
                cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 3)
                sq = w // 8
                for i in range(1, 8):
                    cv2.line(dbg, (x+i*sq, y), (x+i*sq, y+h), (0,200,0), 1)
                    cv2.line(dbg, (x, y+i*sq), (x+w, y+i*sq), (0,200,0), 1)
                ts        = datetime.now().strftime('%H%M%S')
                path_full = f"debug_images/screen_{ts}.jpg"
                path_crop = f"debug_images/crop_{ts}.jpg"
                cv2.imwrite(path_full, dbg)
                pil = self.player.detector.crop_to_pil(img, region)
                pil.save(path_crop)
                self.root.after(0, lambda: self.log(
                    f"Saved: {path_full} + {path_crop}", 'info'
                ))
            else:
                self.root.after(0, lambda: self.log(
                    "No board region ? detect first", 'warning'
                ))
        threading.Thread(target=_run, daemon=True).start()

    def _toggle_debug(self):
        v = self.debug_mode.get()
        self.player.debug = self.player.detector.debug = v
        self.log(f"Debug images: {'ON' if v else 'OFF'}", 'info')

    def _upd_think(self, *_):
        v = self.think_time.get()
        self.think_lbl.config(text=f"{v:.1f}s")
        self.player.think_time = v

    def _upd_speed(self, *_):
        st = self.settle_time.get()
        pi = self.poll_interval.get()
        fr = self.fast_reply_window.get()
        self.settle_lbl.config(text=f"{st:.2f}s")
        self.poll_lbl.config(text=f"{pi:.2f}s")
        self.fast_lbl.config(text=f"{fr:.2f}s")
        self.player.settle_time = st
        self.player.poll_interval = pi
        self.player.change_accept_after = max(0.15, pi * 2.0)
        self.player.fast_reply_window = fr

    def _upd_engine(self, *_):
        threads = max(1, int(round(self.engine_threads.get())))
        h = max(16, int(round(self.engine_hash.get())))
        skill = max(0, min(20, int(round(self.engine_skill.get()))))
        elo = max(1350, min(2850, int(round(self.engine_elo.get()))))
        self.threads_lbl.config(text=str(threads))
        self.hash_lbl.config(text=str(h))
        self.skill_lbl.config(text=str(skill))
        self.elo_lbl.config(text=str(elo))
        self.player.engine_threads = threads
        self.player.engine_hash = h
        self.player.engine_skill = skill
        self.player.engine_elo = elo
        self.player.limit_strength = bool(self.limit_strength.get())
        self.player.ponder_enabled = bool(self.ponder_enabled.get())

    def _reinit_engine(self):
        self._upd_engine()
        try:
            ok = self.player.initialize_engine()
            self.log("Engine reinitialized" if ok else "Engine reinit failed",
                     'success' if ok else 'error')
        except Exception as e:
            self.log(f"Engine reinit failed: {e}", 'error')

    def _start(self):
        if self.is_playing: return
        color = self.player_color.get()
        self.is_playing = True
        self.stop_event.clear()
        self.hist_box.delete(0, tk.END)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.nb.select(0)
        self.log(f"? Starting as {color}", 'success')
        self.auto_thread = threading.Thread(
            target=self.player.play_auto,
            kwargs={'color': color, 'stop_event': self.stop_event},
            daemon=True
        )
        self.auto_thread.start()
        self.root.after(500, self._monitor)

    def _stop(self):
        self.stop_event.set()
        self.player.is_running = False
        self.log("? Stop requested", 'warning')

    def _monitor(self):
        if self.auto_thread and self.auto_thread.is_alive():
            self.root.after(500, self._monitor)
        else:
            self.is_playing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.log("Auto-play finished", 'info')

    def _on_close(self):
        self.stop_event.set()
        self.player.is_running = False
        self.player._quit_engine()
        self.root.after(700, self.root.destroy)


###############################################################################

def main():
    root = tk.Tk()
    ChessAutoPlayerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()