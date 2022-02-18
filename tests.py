import cv2 as cv
from pathlib import Path
from functions import match_template, get_templates


def create_tests(methods):
    templates = get_templates()
    for meth in methods:
        print(meth)
        output_path = 'tests_output/' + meth.split('.')[1].lower()
        Path(output_path).mkdir(parents=True, exist_ok=True)

        for i in range(1, 12):
            test_name = f'test{i}.png'
            img = cv.imread(f'eau_rouge_tests/{test_name}', 0)
            img2 = img.copy()

            match_template(img2, templates, meth, f'{output_path}/{test_name}')