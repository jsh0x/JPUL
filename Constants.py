__author__ = 'jsh0x'
__version__ = '1.0.0'

import numpy as np
from PIL import Image


FORM_LIST = ()

CHARACTER_ARRAYS = {'a': np.array(((14, 38, 62, 72, 57, 33, 9),
								   (14, 107, 153, 140, 136, 73, 28),
								   (17, 64, 112, 145, 122, 90, 48),
								   (24, 56, 129, 132, 103, 95, 57),
								   (44, 116, 82, 88, 98, 87, 58),
								   (51, 108, 95, 89, 90, 91, 58),
								   (30, 81, 135, 135, 122, 114, 43),
								   (10, 34, 58, 72, 68, 45, 21)), dtype=np.uint8), 'A': np.array((), dtype=np.uint8),
                    'b': np.array(((22, 23, 20, 1, 0, 0, 0),
								   (48, 120, 40, 2, 0, 0, 0),
								   (56, 97, 56, 3, 0, 0, 0),
								   (56, 80, 95, 68, 49, 25, 2),
								   (60, 79, 74, 130, 132, 54, 19),
								   (60, 103, 84, 89, 79, 94, 43),
								   (60, 93, 67, 28, 63, 116, 60),
								   (58, 99, 60, 21, 63, 95, 59),
								   (62, 100, 105, 84, 74, 90, 38),
								   (54, 128, 121, 135, 114, 49, 18),
								   (26, 50, 70, 68, 45, 21, 1)), dtype=np.uint8), 'B': np.array((), dtype=np.uint8),
                    'c': np.array(((1, 22, 46, 69, 57, 33, 9),
								   (19, 49, 124, 136, 137, 74, 35),
								   (40, 95, 79, 80, 77, 133, 52),
								   (58, 93, 62, 12, 22, 27, 26),
								   (58, 93, 62, 12, 22, 27, 26),
								   (40, 95, 79, 80, 77, 133, 52),
								   (19, 49, 124, 136, 137, 74, 35),
								   (1, 22, 46, 69, 57, 33, 9)), dtype=np.uint8), 'C': np.array((), dtype=np.uint8),
                    'd': np.array(((0, 0, 0, 4, 22, 23, 20),
								   (0, 0, 0, 7, 48, 120, 40),
								   (0, 0, 0, 10, 56, 97, 56),
								   (4, 26, 50, 77, 94, 89, 57),
								   (22, 55, 129, 126, 100, 91, 58),
								   (43, 94, 78, 90, 90, 91, 58),
								   (58, 93, 62, 24, 59, 96, 57),
								   (58, 93, 62, 28, 63, 92, 57),
								   (43, 94, 78, 90, 78, 86, 57),
								   (22, 55, 129, 129, 100, 102, 42),
								   (4, 26, 50, 67, 63, 40, 20)), dtype=np.uint8), 'D': np.array((), dtype=np.uint8),
                    'e': np.array(((4, 26, 50, 70, 53, 29, 6),
								   (21, 52, 133, 136, 137, 59, 25),
								   (46, 90, 108, 150, 121, 92, 53),
								   (64, 105, 98, 132, 122, 136, 63),
								   (64, 96, 109, 86, 96, 80, 53),
								   (43, 97, 81, 83, 74, 141, 36),
								   (19, 49, 124, 134, 134, 61, 29),
								   (1, 22, 46, 68, 53, 29, 6)), dtype=np.uint8), 'E': np.array((), dtype=np.uint8),
                    'f': np.array(((0, 14, 38, 47, 33),
								   (1, 35, 96, 146, 75),
								   (4, 56, 105, 80, 36),
								   (26, 82, 80, 86, 28),
								   (30, 106, 101, 120, 34),
								   (29, 78, 77, 83, 27),
								   (10, 56, 97, 56, 3),
								   (10, 56, 97, 56, 3),
								   (10, 56, 97, 56, 3),
								   (7, 48, 120, 40, 2),
								   (4, 22, 23, 20, 1)), dtype=np.uint8), 'F': np.array((), dtype=np.uint8),
                    'g': np.array(((4, 26, 50, 70, 68, 45, 21),
								   (22, 55, 129, 132, 125, 118, 42),
								   (43, 94, 78, 87, 94, 89, 57),
								   (58, 93, 62, 22, 56, 97, 56),
								   (58, 93, 62, 22, 56, 97, 56),
								   (43, 94, 78, 87, 94, 89, 57),
								   (22, 55, 129, 126, 102, 95, 56),
								   (26, 72, 119, 148, 102, 88, 43),
								   (26, 150, 146, 138, 136, 57, 22),
								   (22, 46, 69, 71, 53, 29, 6)), dtype=np.uint8), 'G': np.array((), dtype=np.uint8),
                    'h': np.array(((22, 23, 20, 1, 0, 0, 0),
								   (48, 120, 40, 2, 0, 0, 0),
								   (56, 97, 56, 3, 0, 0, 0),
								   (56, 83, 91, 65, 57, 33, 9),
								   (60, 80, 73, 139, 136, 73, 28),
								   (60, 108, 80, 80, 83, 98, 47),
								   (60, 93, 67, 22, 56, 97, 56),
								   (56, 97, 56, 14, 56, 97, 56),
								   (56, 97, 56, 14, 56, 97, 56),
								   (48, 120, 40, 9, 48, 120, 40),
								   (22, 23, 20, 5, 22, 23, 20)), dtype=np.uint8), 'H': np.array((), dtype=np.uint8),
                    'i': np.array(((22, 23, 20),
								   (40, 143, 25),
								   (44, 46, 39),
								   (48, 120, 40),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (48, 120, 40),
								   (22, 23, 20)), dtype=np.uint8), 'I': np.array((), dtype=np.uint8),
                    'j': np.array(((4, 22, 23, 20),
								   (4, 40, 143, 25),
								   (7, 44, 46, 39),
								   (7, 48, 120, 40),
								   (10, 56, 97, 56),
								   (10, 56, 97, 56),
								   (10, 56, 97, 56),
								   (10, 56, 97, 56),
								   (10, 56, 97, 56),
								   (37, 73, 90, 55),
								   (57, 146, 105, 36),
								   (30, 47, 41, 17)), dtype=np.uint8), 'J': np.array((), dtype=np.uint8),
                    'k': np.array(((22, 23, 20, 1, 0, 0, 0),
								   (48, 120, 40, 2, 0, 0, 0),
								   (56, 97, 56, 3, 0, 0, 0),
								   (56, 97, 55, 26, 42, 42, 21),
								   (56, 96, 74, 48, 123, 133, 26),
								   (60, 88, 111, 102, 78, 48, 21),
								   (62, 104, 69, 77, 70, 14, 0),
								   (62, 100, 107, 82, 68, 33, 2),
								   (58, 99, 76, 53, 110, 65, 31),
								   (48, 120, 42, 37, 79, 148, 54),
								   (22, 23, 20, 11, 33, 39, 29)), dtype=np.uint8), 'K': np.array((), dtype=np.uint8),
                    'l': np.array(((22, 23, 20),
								   (48, 120, 40),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (56, 97, 56),
								   (48, 120, 40),
								   (22, 23, 20)), dtype=np.uint8), 'L': np.array((), dtype=np.uint8),
                    'm': np.array(((22, 41, 62, 56, 55, 55, 54, 37, 13),
								   (52, 105, 93, 128, 81, 94, 133, 89, 32),
								   (60, 103, 83, 83, 93, 76, 78, 94, 51),
								   (59, 96, 67, 55, 96, 67, 52, 97, 56),
								   (56, 97, 59, 52, 97, 59, 52, 97, 56),
								   (56, 97, 59, 52, 97, 59, 52, 97, 56),
								   (48, 120, 43, 45, 120, 43, 45, 120, 40),
								   (22, 23, 23, 23, 23, 23, 23, 23, 20)), dtype=np.uint8), 'M': np.array((), dtype=np.uint8),
                    'n': np.array(((22, 37, 57, 63, 57, 33, 9),
								   (52, 103, 86, 140, 136, 73, 28),
								   (60, 108, 80, 80, 83, 98, 47),
								   (60, 93, 67, 22, 56, 97, 56),
								   (56, 97, 56, 14, 56, 97, 56),
								   (56, 97, 56, 14, 56, 97, 56),
								   (48, 120, 40, 9, 48, 120, 40),
								   (22, 23, 20, 5, 22, 23, 20)), dtype=np.uint8), 'N': np.array((), dtype=np.uint8),
                    'o': np.array(((4, 26, 50, 69, 49, 25, 2),
								   (22, 55, 129, 127, 132, 54, 19),
								   (43, 94, 78, 93, 76, 92, 39),
								   (58, 93, 62, 28, 60, 94, 55),
								   (58, 93, 62, 28, 60, 94, 55),
								   (43, 94, 78, 90, 74, 90, 38),
								   (22, 55, 129, 130, 114, 49, 18),
								   (4, 26, 50, 66, 45, 21, 1)), dtype=np.uint8), 'O': np.array((), dtype=np.uint8),
                    'p': np.array(((22, 41, 62, 65, 49, 25, 2),
								   (52, 105, 97, 134, 132, 54, 19),
								   (60, 103, 90, 85, 79, 94, 43),
								   (59, 96, 63, 25, 63, 116, 60),
								   (58, 99, 60, 21, 63, 95, 59),
								   (59, 100, 100, 83, 74, 90, 38),
								   (59, 89, 96, 136, 114, 49, 18),
								   (56, 85, 99, 68, 45, 21, 1),
								   (48, 120, 40, 2, 0, 0, 0),
								   (22, 23, 20, 1, 0, 0, 0)), dtype=np.uint8), 'P': np.array((), dtype=np.uint8),
                    'q': np.array(((4, 26, 50, 70, 68, 45, 21),
								   (21, 52, 133, 133, 122, 114, 43),
								   (42, 95, 77, 86, 90, 91, 58),
								   (57, 97, 58, 21, 59, 96, 57),
								   (60, 98, 62, 21, 59, 96, 57),
								   (49, 113, 81, 85, 86, 89, 57),
								   (28, 63, 135, 131, 82, 83, 57),
								   (6, 30, 54, 76, 90, 84, 56),
								   (0, 0, 0, 7, 48, 120, 40),
								   (0, 0, 0, 4, 22, 23, 20)), dtype=np.uint8), 'Q': np.array((), dtype=np.uint8),
                    'r': np.array(((22, 41, 60, 44, 25),
								   (52, 105, 96, 157, 36),
								   (60, 103, 89, 50, 25),
								   (59, 96, 63, 8, 0),
								   (56, 97, 56, 3, 0),
								   (56, 97, 56, 3, 0),
								   (48, 120, 40, 2, 0),
								   (22, 23, 20, 1, 0)), dtype=np.uint8), 'R': np.array((), dtype=np.uint8),
                    's': np.array(((10, 34, 58, 60, 37, 13),
								   (31, 80, 131, 127, 91, 36),
								   (49, 98, 118, 102, 122, 43),
								   (40, 110, 121, 76, 79, 49),
								   (44, 73, 96, 73, 131, 50),
								   (46, 137, 82, 117, 80, 61),
								   (35, 76, 129, 131, 86, 35),
								   (10, 34, 58, 60, 37, 13)), dtype=np.uint8), 'S': np.array((), dtype=np.uint8),
                    't': np.array(((4, 22, 23, 20, 1),
								   (7, 48, 120, 40, 2),
								   (25, 74, 72, 78, 23),
								   (26, 88, 108, 102, 24),
								   (25, 74, 72, 78, 23),
								   (10, 56, 97, 56, 3),
								   (10, 56, 97, 56, 3),
								   (8, 61, 90, 80, 27),
								   (4, 45, 126, 140, 35),
								   (1, 22, 45, 46, 25)), dtype=np.uint8), 'T': np.array((), dtype=np.uint8),
                    'u': np.array(((22, 23, 20, 5, 22, 23, 20),
								   (48, 120, 40, 9, 48, 120, 40),
								   (56, 97, 56, 14, 56, 97, 56),
								   (56, 97, 56, 14, 56, 97, 56),
								   (60, 100, 58, 18, 59, 96, 57),
								   (52, 109, 91, 82, 86, 89, 57),
								   (31, 80, 135, 138, 104, 106, 42),
								   (10, 34, 58, 68, 63, 40, 20)), dtype=np.uint8), 'U': np.array((), dtype=np.uint8),
                    'v': np.array(((26, 31, 28, 9, 26, 31, 28),
								   (50, 140, 60, 32, 56, 140, 61),
								   (43, 71, 69, 69, 65, 82, 50),
								   (18, 62, 90, 94, 77, 61, 24),
								   (4, 49, 86, 62, 88, 54, 7),
								   (0, 31, 70, 93, 74, 38, 1),
								   (0, 13, 63, 124, 70, 18, 0),
								   (0, 4, 26, 31, 28, 6, 0)), dtype=np.uint8), 'V': np.array((), dtype=np.uint8),
                    'w': np.array(((29, 31, 26, 23, 23, 23, 24, 27, 26),
								   (64, 152, 51, 56, 127, 56, 52, 136, 60),
								   (60, 88, 57, 54, 115, 56, 66, 102, 58),
								   (38, 66, 40, 50, 52, 63, 53, 64, 41),
								   (21, 57, 38, 65, 87, 69, 43, 56, 24),
								   (8, 58, 92, 59, 78, 66, 107, 60, 11),
								   (1, 41, 103, 69, 42, 66, 102, 42, 2),
								   (0, 18, 27, 26, 15, 23, 23, 17, 0)), dtype=np.uint8), 'W': np.array((), dtype=np.uint8),
                    'x': np.array(((14, 34, 35, 24, 27, 35, 31, 9),
								   (14, 89, 116, 66, 61, 141, 67, 9),
								   (14, 48, 67, 57, 70, 62, 40, 9),
								   (0, 18, 73, 64, 58, 64, 10, 0),
								   (0, 22, 69, 78, 69, 63, 14, 0),
								   (18, 55, 80, 68, 68, 66, 48, 13),
								   (20, 105, 93, 58, 57, 130, 85, 13),
								   (18, 35, 34, 18, 22, 35, 34, 13)), dtype=np.uint8), 'X': np.array((), dtype=np.uint8),
                    'y': np.array(((14, 31, 31, 17, 14, 27, 27, 13),
								   (17, 90, 103, 42, 37, 88, 83, 15),
								   (17, 66, 113, 62, 65, 109, 64, 15),
								   (4, 45, 73, 50, 55, 93, 45, 2),
								   (0, 21, 60, 59, 53, 55, 24, 0),
								   (0, 7, 58, 86, 76, 51, 7, 0),
								   (0, 2, 57, 72, 72, 36, 1, 0),
								   (6, 31, 75, 103, 60, 15, 0, 0),
								   (6, 60, 151, 92, 35, 2, 0, 0),
								   (6, 30, 43, 37, 13, 0, 0, 0)), dtype=np.uint8), 'Y': np.array((), dtype=np.uint8),
                    'z': np.array(((14, 38, 62, 71, 49, 25),
								   (14, 107, 143, 118, 132, 49),
								   (14, 41, 96, 84, 88, 44),
								   (0, 22, 57, 113, 57, 19),
								   (10, 48, 87, 68, 33, 2),
								   (28, 75, 64, 114, 62, 29),
								   (27, 102, 123, 127, 159, 56),
								   (18, 41, 65, 71, 53, 29)), dtype=np.uint8), 'Z': np.array((), dtype=np.uint8),
                    '1': np.array(((0, 1, 21, 23, 22),
								   (18, 43, 84, 122, 50),
								   (20, 127, 108, 112, 63),
								   (18, 44, 102, 99, 63),
								   (0, 3, 61, 112, 61),
								   (0, 3, 61, 112, 61),
								   (0, 3, 61, 112, 61),
								   (0, 3, 61, 112, 61),
								   (0, 3, 61, 112, 61),
								   (0, 2, 43, 136, 47),
								   (0, 1, 21, 23, 22)), dtype=np.uint8), '!': np.array((), dtype=np.uint8),
                    '2': np.array(((0, 14, 38, 62, 65, 41, 17, 0),
								   (10, 41, 86, 136, 132, 100, 43, 9),
								   (10, 70, 101, 79, 91, 80, 57, 22),
								   (10, 27, 27, 18, 46, 49, 58, 30),
								   (0, 0, 0, 10, 57, 82, 59, 23),
								   (0, 0, 10, 40, 65, 99, 47, 10),
								   (0, 10, 40, 58, 96, 49, 22, 1),
								   (6, 39, 64, 98, 50, 22, 1, 0),
								   (20, 70, 87, 114, 95, 62, 37, 13),
								   (20, 96, 122, 118, 141, 154, 100, 13),
								   (14, 38, 62, 72, 72, 60, 37, 13)), dtype=np.uint8), '@': np.array((), dtype=np.uint8),
                    '3': np.array(((0, 14, 38, 62, 57, 33, 9, 0),
								   (6, 40, 92, 132, 136, 76, 31, 2),
								   (6, 58, 121, 80, 91, 110, 52, 8),
								   (6, 26, 27, 26, 62, 95, 65, 10),
								   (0, 0, 10, 38, 74, 115, 53, 8),
								   (0, 0, 10, 79, 129, 66, 59, 11),
								   (0, 0, 10, 34, 69, 88, 49, 22),
								   (10, 27, 27, 18, 46, 49, 58, 30),
								   (10, 70, 101, 79, 87, 84, 53, 22),
								   (10, 41, 86, 136, 136, 83, 39, 9),
								   (0, 14, 38, 62, 60, 37, 13, 0)), dtype=np.uint8), '#': np.array((), dtype=np.uint8),
                    '4': np.array(((0, 0, 0, 14, 27, 27, 13),
								   (0, 0, 4, 40, 77, 75, 26),
								   (0, 0, 21, 68, 92, 52, 38),
								   (0, 7, 44, 72, 36, 54, 38),
								   (1, 28, 63, 60, 36, 47, 38),
								   (11, 49, 113, 72, 33, 47, 38),
								   (29, 73, 72, 127, 65, 54, 62),
								   (29, 106, 127, 111, 98, 109, 75),
								   (18, 42, 65, 91, 68, 54, 62),
								   (0, 0, 0, 20, 55, 70, 26),
								   (0, 0, 0, 10, 23, 23, 13)), dtype=np.uint8), '$': np.array((), dtype=np.uint8),
                    '5': np.array(((1, 22, 46, 69, 71, 53, 29, 6),
								   (2, 44, 127, 123, 142, 159, 56, 6),
								   (5, 63, 99, 102, 74, 53, 29, 6),
								   (8, 64, 92, 107, 64, 37, 13, 0),
								   (13, 63, 105, 104, 134, 83, 39, 9),
								   (10, 63, 115, 90, 87, 84, 53, 22),
								   (6, 26, 27, 21, 39, 44, 53, 34),
								   (10, 27, 27, 18, 41, 50, 59, 31),
								   (10, 70, 101, 79, 89, 107, 56, 18),
								   (10, 41, 86, 136, 133, 89, 39, 6),
								   (0, 14, 38, 62, 60, 37, 13, 0)), dtype=np.uint8), '%': np.array((), dtype=np.uint8),
                    '6': np.array(((0, 6, 30, 54, 65, 41, 17, 0),
								   (1, 28, 65, 131, 128, 98, 44, 13),
								   (7, 53, 108, 82, 88, 84, 82, 13),
								   (17, 67, 96, 67, 24, 27, 27, 13),
								   (26, 57, 66, 101, 64, 41, 17, 0),
								   (30, 53, 61, 104, 132, 100, 43, 9),
								   (26, 53, 85, 93, 90, 80, 57, 22),
								   (17, 62, 71, 57, 49, 49, 58, 30),
								   (7, 45, 117, 82, 92, 80, 57, 22),
								   (1, 27, 66, 138, 130, 100, 43, 9),
								   (0, 6, 30, 54, 65, 41, 17, 0)), dtype=np.uint8), '^': np.array((), dtype=np.uint8),
                    '7': np.array(((10, 34, 58, 72, 72, 71, 49, 25),
								   (10, 83, 157, 143, 140, 125, 140, 48),
								   (10, 34, 58, 72, 88, 91, 85, 46),
								   (0, 0, 0, 4, 39, 74, 54, 22),
								   (0, 0, 0, 13, 51, 106, 46, 7),
								   (0, 0, 1, 34, 63, 60, 31, 1),
								   (0, 0, 11, 53, 118, 52, 11, 0),
								   (0, 0, 28, 63, 70, 41, 2, 0),
								   (0, 7, 50, 84, 56, 18, 0, 0),
								   (0, 7, 58, 106, 39, 6, 0, 0),
								   (0, 6, 23, 23, 17, 0, 0, 0)), dtype=np.uint8), '&': np.array((), dtype=np.uint8),
                    '8': np.array(((0, 10, 34, 58, 60, 37, 13, 0),
								   (4, 35, 79, 133, 130, 89, 39, 6),
								   (7, 53, 114, 91, 93, 106, 56, 14),
								   (8, 60, 99, 61, 66, 78, 65, 17),
								   (4, 42, 95, 76, 87, 114, 48, 11),
								   (4, 43, 58, 111, 114, 62, 50, 8),
								   (13, 45, 103, 78, 77, 91, 45, 18),
								   (20, 61, 68, 54, 48, 52, 61, 27),
								   (16, 56, 99, 91, 91, 80, 57, 22),
								   (6, 40, 92, 132, 131, 100, 43, 9),
								   (0, 14, 38, 62, 65, 41, 17, 0)), dtype=np.uint8), '*': np.array((), dtype=np.uint8),
                    '9': np.array(((0, 14, 38, 62, 57, 33, 9, 0),
								   (6, 40, 92, 132, 136, 76, 31, 2),
								   (16, 56, 99, 91, 84, 115, 48, 11),
								   (22, 60, 66, 56, 51, 58, 58, 24),
								   (16, 56, 99, 91, 89, 68, 53, 34),
								   (6, 40, 92, 132, 116, 51, 47, 38),
								   (0, 14, 38, 62, 95, 49, 51, 34),
								   (10, 27, 27, 24, 60, 84, 65, 24),
								   (10, 70, 101, 85, 84, 117, 58, 11),
								   (10, 41, 86, 130, 129, 74, 34, 2),
								   (0, 14, 38, 62, 57, 33, 9, 0)), dtype=np.uint8), '(': np.array((), dtype=np.uint8),
                    '0': np.array(((0, 14, 38, 62, 65, 41, 17, 0),
								   (4, 36, 93, 133, 132, 102, 39, 6),
								   (10, 55, 93, 94, 95, 80, 57, 14),
								   (20, 54, 86, 50, 42, 63, 48, 27),
								   (26, 46, 55, 42, 34, 40, 49, 34),
								   (30, 38, 47, 38, 30, 38, 47, 38),
								   (26, 46, 55, 42, 34, 40, 49, 34),
								   (20, 54, 86, 50, 42, 63, 48, 27),
								   (10, 55, 93, 94, 95, 80, 57, 14),
								   (4, 36, 93, 133, 132, 102, 39, 6),
								   (0, 14, 38, 62, 65, 41, 17, 0)), dtype=np.uint8), ')': np.array((), dtype=np.uint8),
                    '-': np.array(((0, 0, 0, 0, 0),
								   (14, 38, 55, 41, 17),
								   (14, 107, 160, 125, 17),
								   (14, 38, 55, 41, 17),
								   (0, 0, 0, 0, 0)), dtype=np.uint8), '_': np.array((), dtype=np.uint8),
                    '=': np.array(((30, 54, 72, 72, 57, 33),
								   (61, 161, 143, 143, 158, 77),
								   (60, 108, 143, 143, 113, 66),
								   (61, 161, 143, 143, 158, 77),
								   (30, 54, 72, 72, 57, 33)), dtype=np.uint8), '+': np.array((), dtype=np.uint8),
                    '[': np.array(((6, 30, 50, 45, 21),
								   (12, 59, 142, 124, 26),
								   (18, 58, 76, 78, 21),
								   (18, 51, 83, 51, 0),
								   (18, 51, 83, 51, 0),
								   (18, 51, 83, 51, 0),
								   (18, 51, 83, 51, 0),
								   (18, 51, 83, 51, 0),
								   (18, 51, 83, 51, 0),
								   (18, 51, 83, 51, 0),
								   (18, 58, 76, 78, 21),
								   (12, 59, 142, 124, 26),
								   (6, 30, 50, 45, 21)), dtype=np.uint8), '{': np.array((), dtype=np.uint8),
                    ']': np.array(((18, 42, 50, 33, 9),
								   (20, 114, 142, 64, 18),
								   (18, 70, 56, 52, 27),
								   (0, 42, 57, 42, 27),
								   (0, 42, 57, 42, 27),
								   (0, 42, 57, 42, 27),
								   (0, 42, 57, 42, 27),
								   (0, 42, 57, 42, 27),
								   (0, 42, 57, 42, 27),
								   (0, 42, 57, 42, 27),
								   (18, 70, 56, 52, 27),
								   (20, 114, 142, 64, 18),
								   (18, 42, 50, 33, 9)), dtype=np.uint8), '}': np.array((), dtype=np.uint8),
                    ';': np.array(((21, 23, 22),
								   (26, 159, 34),
								   (21, 23, 22),
								   (0, 0, 0),
								   (0, 0, 0),
								   (21, 23, 22),
								   (40, 140, 45),
								   (40, 119, 42),
								   (18, 19, 18)), dtype=np.uint8), ':': np.array((), dtype=np.uint8),
                    "'": np.array(((10, 23, 23, 13),
								   (20, 55, 70, 26),
								   (26, 40, 54, 34),
								   (16, 44, 52, 22),
								   (6, 15, 15, 9)), dtype=np.uint8), '"': np.array((), dtype=np.uint8),
                    ',': np.array(((0, 0, 0),
								   (21, 23, 22),
								   (40, 140, 45),
								   (40, 119, 42),
								   (18, 19, 18)), dtype=np.uint8), '<': np.array((), dtype=np.uint8),
                    '.': np.array(((0, 0, 0),
								   (21, 23, 22),
								   (26, 159, 34),
								   (21, 23, 22)), dtype=np.uint8), '>': np.array((), dtype=np.uint8),
                    '/': np.array(((0, 0, 0, 0, 17, 19, 19),
								   (0, 0, 0, 7, 36, 110, 38),
								   (0, 0, 0, 24, 55, 73, 34),
								   (0, 0, 7, 42, 92, 48, 15),
								   (0, 0, 24, 55, 73, 34, 2),
								   (0, 7, 42, 92, 48, 15, 0),
								   (0, 24, 55, 73, 34, 2, 0),
								   (7, 42, 92, 48, 15, 0, 0),
								   (28, 58, 77, 34, 2, 0, 0),
								   (30, 140, 44, 15, 0, 0, 0),
								   (21, 23, 22, 2, 0, 0, 0)), dtype=np.uint8), '?': np.array((), dtype=np.uint8)}


def _get():
	retval = ""
	im = Image.open('char.png').convert('L')
	img = np.array(im, dtype=np.uint8)
	#pixel = im.load()
	#for y in range(im.size[1]):
		#string = "'{}', "*im.size[0]
		#string = "("+string.rstrip(', ')+"),\n"
		#temp_list = []
		#for x in range(im.size[0]):
			#r,g,b = pixel[x,y]
			#rgb = hex(r).split('x')[1].rjust(2,'0')+hex(g).split('x')[1].rjust(2,'0')+hex(b).split('x')[1].rjust(2,'0')
			#temp_list.append(rgb)
		#retval+= string.format(*temp_list)
	#print(retval)
	for row in img:
		print("("+str(row.tolist()).lstrip("[").rstrip("]")+"),")
_get()