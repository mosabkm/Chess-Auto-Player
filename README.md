# Chess-Auto-Player
A Python-based chess auto-player with a graphical user interface (GUI), screen-based chessboard detection, FEN position recognition, and Stockfish engine integration.

Version: 4.3.0

Features

- Automatic board detection — identifies the chessboard using grid analysis and contour detection.
- FEN recognition — converts board images into chess positions using "chessimg2pos", with multiple image preprocessing methods and voting.
- Stockfish integration — calculates chess moves using the Stockfish chess engine.
- Automatic move execution — uses mouse control to play calculated moves on the detected board.
- Opponent move detection — monitors board changes and uses repeated readings to detect the opponent's moves.
- Configurable engine settings — adjust thinking time, threads, hash memory, skill level, and playing strength.
- Graphical interface — manage gameplay, board detection, engine settings, move history, and logs through Tkinter.
- Debugging tools — save board images and inspect application logs.

Requirements

- Python 3.10 or newer recommended
- Stockfish chess engine
- Tkinter
- Pillow
- NumPy
- OpenCV
- python-chess
- chessimg2pos
- PyAutoGUI

Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/chess-auto-player.git
cd chess-auto-player

Replace "YOUR_USERNAME" with your GitHub username.

2. Install Python dependencies

python -m pip install Pillow numpy opencv-python python-chess chessimg2pos pyautogui

3. Install Stockfish

Download Stockfish for your operating system from:

https://stockfishchess.org/download/

Make sure the executable is accessible and configure its path in the application if it is not detected automatically.

4. Run the application

python3 chess_auto_player.py

Usage

1. Launch the application.
2. Verify that Stockfish is detected.
3. Open a chessboard on your screen.
4. Use automatic board detection or enter the board region manually.
5. Select your playing color.
6. Adjust the engine and detection settings as needed.
7. Start auto-play and monitor the activity log and move history.
8. Use Stop to end the session.

Troubleshooting

- Stockfish not found: Select the correct executable using the Browse button.
- Board detection fails: Adjust the board region manually.
- FEN recognition fails: Check board visibility and image quality, and confirm that "chessimg2pos" is installed.
- Mouse control fails: Check operating-system permissions and screen accessibility.
- Performance issues: Reduce the engine thread count or hash allocation.

Disclaimer

This project is provided for educational and experimental purposes. Automated play may violate the rules of online chess platforms. Use it only where permitted, such as local testing or authorized environments.

License

No license has been specified yet. All rights remain with the copyright holder unless a license is added to this repository.
