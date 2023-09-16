from extraction import get_boxes

def handwritten_text_recognition(img, debug=False):

    boxes = get_boxes(img, debug=debug)
    return None