from PIL import Image
import numpy as np
import time
import pytesseract
pytesseract.get_tesseract_version()


def Levenshtein_Distance(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def cal_cer_ed(path_ours, tail='_rec'):
    print(path_ours, 'start')
    print(f"started at {time.strftime('%H:%M:%S')}")
    path_gt = './GT/'
    N = 196
    cer1 = []
    ed1 = []
    check = [0 for _ in range(N + 1)]
    # img index in UDIR test set for OCR evaluation
    lis = [2, 5, 17, 19, 20, 23, 31, 37, 38, 39, 40, 41, 43, 45, 47, 48, 51, 54, 57, 60, 61, 62, 64, 65, 67, 68, 70, 75,
           76, 77, 78, 80, 81, 83, 84, 85, 87, 88, 90, 91, 93, 96, 99, 100, 101, 102, 103, 104, 105, 134, 137, 138, 140,
           150, 151, 155, 158, 162, 163, 164, 165, 166, 169, 170, 172, 173, 175, 177, 178, 182, ]
    for i in range(1, N):
        if i not in lis:
            continue
        gt = Image.open(path_gt + str(i) + '.png')
        img1 = Image.open(path_ours + str(i) + tail)
        content_gt = pytesseract.image_to_string(gt)
        content1 = pytesseract.image_to_string(img1)
        l1 = Levenshtein_Distance(content_gt, content1)
        ed1.append(l1)
        cer1.append(l1 / len(content_gt))
        check[i] = cer1[-1]

    CER = np.mean(cer1)
    ED = np.mean(ed1)
    print(f"finished at {time.strftime('%H:%M:%S')}")
    return [path_ours, CER, ED]
