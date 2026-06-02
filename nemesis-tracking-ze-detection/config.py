# Arena Config

ARENA_WIDTH_FT = 48
ARENA_HEIGHT_FT = 48

TILES_PER_SIDE = 12
TILE_SIZE_FT = 4
SUBDIVISIONS = 2
GRID_CELLS_PER_SIDE = TILES_PER_SIDE * SUBDIVISIONS
CELL_SIZE_FT = TILE_SIZE_FT / SUBDIVISIONS

# PHYSICS

MAX_ROBOT_SPEED_FT_PER_SEC = 30.0

# Calibration for homography (we may need to fix these/ add more for better precision)


CALIBRATION_CLICKS = [
    (518, 554),  # bottom-left
    (551, 424),  # top-left
    (724, 433),  # top-right
    (720, 566),  # bottom-right
]

CALIBRATION_TILE_X = 5  # tiles from left
CALIBRATION_TILE_Y = 2  # tiles from bottom
CALIBRATION_SQUARE_SIZE_FT = 4


# Detection config (we need to run tests and figure out what is best for these)

MIN_CONFIDENCE = 0.3
MIN_BOX_AREA_PX = 100
MAX_BOX_AREA_PX = 150000
MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 10.0

# Tracking (once again also we need to mess with these)


MAX_DETECTION_DIST_FT = 50.0
CLOSE_PROXIMITY_FT = 3.0


# perf mode

HIGH_PERFORMANCE_MODE = False

# viz config

SIM_SCALE = 15  # Pixels per foot in simulation view
ROBOT_SIZE_FT = 3  # Size of robot rectangle in simulation


# Colors (BGR)
COLORS = {
    "USC": (0, 0, 255),  # Red
    "ENEMY": (235, 160, 50),  # Blue
    "ACCENT_BLUE": (235, 160, 50),
    "ACCENT_CYAN": (210, 180, 80),
    "TEXT_WHITE": (240, 240, 240),
    "TEXT_DIM": (140, 140, 140),
    "BORDER": (60, 60, 60),
    "GRID": (180, 130, 70),
    "GRID_MAJOR": (200, 150, 90),
    "HAZARD": (40, 40, 80),
}
