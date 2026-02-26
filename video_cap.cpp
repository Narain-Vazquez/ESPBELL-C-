#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("http://192.168.0.50:8082/stream"); // Open the default camera
    if (!cap.isOpened()) {
        printf("Error: Cannot access the camera\n");
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // Capture a frame
        if (frame.empty()) break;

        cv::imshow("Captured Frame", frame); // Display the frame
        if (cv::waitKey(30) >= 0) break; // Exit on key press
    }

    cap.release(); // Release the camera
    return 0;
}
