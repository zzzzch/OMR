//
// Created by zch on 21-4-19.
//

#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>

//#include "base_io.h"
//#include "tictoc.hpp"
//
//#include "charging_gun_detect.h"
//#include "pcl_tools.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


//using namespace zjlab;
//using namespace pose;

DEFINE_string(image_path,
              "/home/zhachanghai/charging_gun/rgb_image/", "Input rgb image file path");

DEFINE_string(target_path,
              "/home/zhachanghai/charging_gun/target.png", "target image file path");

using namespace std;
using namespace cv;

typedef enum {NOTEHEAD=0, QUARTERREST=1, EIGHTHREST=2} Type;

class DetectedSymbol
{
public:
    int row, col, width, height;
    Type type;
    char pitch;
    double confidence;
};


class Dimensions
{
public:
    int row_coordinate;
    int spacing;
    bool treble;
};


void split_music_paragraph(vector<Mat>& music_paragraph,Mat &origin){
    vector<int> image_rows;

    for (int i = 0; i < origin.rows; i++){
        int count_image = 0;
        for (int j = 0; j < origin.cols; j++){
            if(origin.at<Vec3b>(i,j)[0] < 20 && origin.at<Vec3b>(i,j)[1] < 20 && origin.at<Vec3b>(i,j)[2] < 20){
                count_image++;
            }
        }
        if(count_image > 0){
            image_rows.push_back(i);
        }
    }

}

void staff_detect(Mat &origin,Mat &result){

    Mat contours;
    origin.copyTo(result);
    Canny(origin,contours,125,350);
    GaussianBlur(contours, contours, Size(3, 5), 1, 2);

//    cvtColor( contours, result, COLOR_GRAY2BGR);
    vector<Vec4i> lines;
    HoughLinesP(contours, lines, 1, CV_PI/180, 20, 500, 10);
    cout << "line size is " << lines.size() << endl;

    sort(lines.begin(), lines.end(),
         [](const Vec4i& a, const Vec4i& b) {
             return a[1] < b[1];
         });

    vector<Vec4i> process_lines;
    int left_line = 3000;
    int right_line = 0;

    for (int i = 0;i < lines.size(); i++){
        Vec4i l = lines[i];
        if(abs(l[3] - l[1]) < 20){
            if(process_lines.size() < 1){
                process_lines.push_back(l);
                left_line = l[0] < left_line ? l[0] : left_line;
                right_line = l[2] > right_line ? l[2] : right_line;

                continue;
            }

            if(l[1] - process_lines.back()[1] > 8){
                process_lines.push_back(l);
                left_line = l[0] < left_line ? l[0] : left_line;
                right_line = l[2] > right_line ? l[2] : right_line;

            }
        }
    }

    cout << "process line size is " << process_lines.size() << endl;
    cout << "left line is " << left_line << "  right line is " << right_line << endl;


    for( size_t i = 0; i < process_lines.size(); i++ )
    {
        Vec4i l = process_lines[i];
        line(result, Point(left_line, l[1]), Point(right_line, l[3]), Scalar(255,0,0), 1);


    }

}


//vector<Dimensions> findStaff(const SDoublePlane &input)
//{
//    int inp_row = input.rows();
//    int inp_col = input.cols();
//    int accum_d = inp_row/10;
//    int accum_r = inp_row-10;
//    vector<Dimensions> staves;
//    _DTwoDimArray<int> accum(accum_r, accum_d);
//
//    for(int i=0;i<accum_r;i++)
//    {
//        for(int j=0;j<accum_d;j++)
//        {
//            accum[i][j] = 0;
//        }
//    }
//    //Voting the Accumulator Space based on the pixel values.
//    for(int i=1;i<accum_r;i++)
//    {
//        for(int j=0;j<inp_col;j++)
//        {
//            for(int h=1;h<=accum_d;h++)
//            {
//                if ((i +(4*h)+1) < inp_row-1)
//                {
//                    if ((input[i][j] == 0 || input[i+1][j] ==0 || input[i-1][j] ==0) && \
//                        (input[i+h][j] == 0 || input[i+h+1][j] == 0 || input[i+h-1][j] == 0) && \
//                        (input[i+(2*h)][j] == 0 || input[i+(2*h)+1][j] == 0 || input[i+(2*h)-1][j] == 0) && \
//                        (input[i+(3*h)][j] == 0 || input[i+(3*h)+1][j] == 0 || input[i+(3*h)-1][j] == 0) && \
//                        (input[i+(4*h)][j] == 0 || input[i+(4*h)+1][j] == 0 || input[i+(4*h)-1][j] == 0))
//                    {
//                        accum[i][h]++;
//                    }
//                }
//            }
//        }
//    }
//    //Finding the Parameters surpassing a threshold
//    Dimensions dim,prev_dim;
//    bool trebble = 1;
//    for(int i=0;i<accum_r;i++)
//    {
//        for(int j=0;j<accum_d;j++)
//        {
//            if (accum[i][j] >= 0.9*inp_col)
//            {
//                dim.row_coordinate =i;
//                dim.spacing = j;
//                dim.trebble = trebble;
//
//                if ((abs(prev_dim.row_coordinate - dim.row_coordinate) > 5) && (dim.spacing > 3))
//                {
//                    staves.push_back(dim);
//                    trebble = !trebble;
//                }
//                //cout << dim.row_coordinate << " " << dim.spacing << " " << abs(prev_dim.row_coordinate - dim.row_coordinate) << endl;
//                prev_dim.row_coordinate = dim.row_coordinate;
//                prev_dim.spacing = dim.spacing;
//                break;
//            }
//        }
//    }
//    return staves;
//}


int main(int argc, char **argv) {

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

    int count = 0;

//    Mat origin = imread("/home/zch/Optical_Music_Recognition/music3.png");
    Mat origin = imread("/home/zch/music_images/485_image.png");

//    Mat origin = imread(FLAGS_image_path + std::to_string(count) + "_image.jpg");
    Mat staff_image;


    vector<Mat> music_paragraph;

//    split_music_paragraph(music_paragraph, origin);

    staff_detect(origin, staff_image);
    imshow("origin",origin);

    imshow("result",staff_image);

    waitKey(0);



    return 0;

}

