from flask import Flask, send_file
import cv2
import io

app = Flask(__name__)


@app.route('/')
def hello_world():
    img = cv2.imread("images/test_image.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    # canny = cv2.Canny(blur, 10, 50)

    # find contours
    contours, _ = cv2.findContours(
        blur,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    _, buf = cv2.imencode(".png", img)
    io_buf = io.BytesIO(buf)
    return send_file(io_buf, mimetype='image/png')
