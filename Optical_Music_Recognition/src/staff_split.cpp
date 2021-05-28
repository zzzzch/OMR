#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

struct Line{
    int left_x;
    int left_y;
    int right_x;
    int right_y;
    int thicknesses;
};

void getStaffLines(int width, int height, Mat& input_image, float threshold, vector<int>& staff_lines, vector<int>& staff_lines_thicknesses){
    vector<int> inital_lines;
    float row_histogram[height] = { 0 };

    for(int i = 0;i < height; i++){
        for (int j = 0; j < width; j++){
            if(input_image.at<uchar>(i,j) < 50){
                row_histogram[i] += 1;
            }
        }
    }

    for (int i = 0;i < height;i++){
        if (row_histogram[i] >= width * threshold){
            inital_lines.push_back(i);
//            LOG(INFO) << " row histogram is " << row_histogram[i] << " height is " << i;
        }
    }


    int cur_thickness = 1;

    for(int i = 0;i < inital_lines.size(); i++){
        if (cur_thickness == 1){
            staff_lines.push_back(inital_lines[i]);
        }
        if (i == int(inital_lines.size() - 1)){
            staff_lines_thicknesses.push_back(cur_thickness);
        }else if (inital_lines[i] + 1 == inital_lines[i + 1]){
            cur_thickness +=1;
        }else {
            staff_lines_thicknesses.push_back(cur_thickness);
            cur_thickness = 1;
        }
    }
}

Mat removeSingleLine(int line_thickness, int line_start, Mat &input, int width){
    Mat output;
    input.copyTo(output);
//    LOG(INFO) << "line_start is " << line_start;

    int line_end = line_start + line_thickness - 1;
    for(int col = 0; col < width; col++){
        if(output.at<uchar>(line_start,col) < 50 || output.at<uchar>(line_end,col) < 50){

            if(output.at<uchar>(line_start-1,col) > 200 && output.at<uchar>(line_end+1,col) > 200){
                for(int j = 0; j < line_thickness; j++){
                    output.at<uchar>(line_start+j,col) = 255;
                }
            }else if(output.at<uchar>(line_start-1,col) > 200 && output.at<uchar>(line_end+1,col) < 50){
                int thick = line_thickness + 1;
                thick = thick < 1 ? 1 : thick;
                for(int j = 0; j < thick; j++){
                    output.at<uchar>(line_start+j,col) = 255;
                }
            }else if(output.at<uchar>(line_start-1,col) < 50 && output.at<uchar>(line_end+1,col) > 200){
                if(col > 0 && col < width - 1 && output.at<uchar>(line_start - 1,col - 1) > 200 && output.at<uchar>(line_start - 1,col + 1) > 200){
                    int thick = line_thickness + 1;
                    thick = thick < 1 ? 1 : thick;
                    for (int j = 0; j < thick; j++){
                        output.at<uchar>(line_start-j,col) = 255;
                    }
                }
            }
        }
    }

    return output;

}


Mat removeStaffLines(Mat& input,int width, vector<int>& staff_lines, vector<int>& staff_lines_thicknesses){
    for(int i = 0; i < staff_lines.size(); i ++){
        int line_start = staff_lines[i];
        int line_thickness = staff_lines_thicknesses[i];
        input = removeSingleLine(line_thickness, line_start, input, width);
    }
    return input;
}

void cutImageIntoBuckets(Mat &input,vector<int>& staff_lines,vector<int>& cutting_position,vector<Mat>& cutted_images){
    int lst_slice = 350;
    int no_of_buckets = staff_lines.size() / 5;
    LOG(INFO) << "no of buckets is " << no_of_buckets;
    for(int i = 0;i < no_of_buckets - 1;i++){
        int start = staff_lines[i * 5 + 4];
        int end = staff_lines[i * 5 + 5];
        int mid_row = (start + end)/2;
        cutting_position.push_back(lst_slice);
        // rect image
        Mat cut_image = input(Rect(0, lst_slice, input.cols, mid_row - lst_slice));
        cutted_images.push_back(cut_image);
        lst_slice = mid_row;
    }
    cutting_position.push_back(lst_slice);
    Mat cut_image = input(Rect(0, lst_slice, input.cols, input.rows - lst_slice));
    cutted_images.push_back(cut_image);
}

void getRefLines(vector<int>& cutting_position,vector<int>& staff_lines,vector<int>& ref_lines,vector<int>& lines_spacing){

    int no_of_buckets = staff_lines.size()/5;

    for(int i = 0; i < no_of_buckets;i++){
        int line_spacing = (staff_lines[i * 5 + 4] - staff_lines[i * 5])/4;
        lines_spacing.push_back(line_spacing);
//        LOG(INFO) << "line_spaceing is " << line_spacing;
        int ref_line = staff_lines[i * 5 + 4] - line_spacing * 1.5;
//        LOG(INFO) << "ref_line is " << ref_line;
        ref_lines.push_back(ref_line);
    }
}

vector<Rect> segmentation(int height_before,Mat& input){
    Mat blurred,thresh,kernel,dilates;
    GaussianBlur(input, blurred, Size(3, 5), 1, 2);
    threshold(blurred, thresh, 120, 255, THRESH_BINARY_INV);

    kernel = getStructuringElement(MORPH_RECT, Size(2,2));
    dilate(thresh,dilates,kernel);
    Rect cnts;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Point> cnt;
    findContours(dilates,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    // TODO: know why to do it & how to deal the params cnt
    cnt = contours.size() == 2 ? contours[0] : contours[1];
    int threshold_min_area = 0;
    int threshold_max_area = input.rows * input.cols;
    vector<Rect> symbols;
    for(int i = 0; i < contours.size(); i ++){
        Rect bound_rect;
        bound_rect = boundingRect(contours[i]);
        double area = contourArea(contours[i]);
        if (area > threshold_min_area && area < threshold_max_area){
            symbols.push_back(Rect(bound_rect.x, bound_rect.y+height_before, bound_rect.width, bound_rect.height+height_before));
        }
    }

    return symbols;
}

void cleanAndCut(Mat& image){
    bitwise_not(image, image);
    std::vector<std::vector<cv::Point> >contours;
    findContours(image,contours,RETR_LIST,CHAIN_APPROX_SIMPLE);
    Mat mask = Mat::ones(image.cols,image.rows,CV_8UC1) * 255;
    sort(contours.begin(), contours.end(),
         [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
             return (contourArea(a) > contourArea(b));
         });
    for(int i = 0;i < contours.size()-1;i++){
        cv::drawContours(image, contours[i], 0, cv::Scalar(0, 0,255), 3);
    }
    bitwise_and(image, image, image, mask);

}

//def clean_and_cut(img):
//img[img > 200] = 255
//img[img <= 200] = 0
//
//img = 255 - img
//
//contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
//mask = np.ones((img.shape[0], img.shape[1]), dtype="uint8") * 255
//contours = sorted(contours, key=cv2.contourArea)
//
//for i in range(len(contours) - 1):
//cv2.drawContours(mask, [contours[i]], -1, 0, -1)
//
//img = cv2.bitwise_and(img, img, mask=mask)
//
//white = np.argwhere(img == 255)
//
//x, y, w, h = cv2.boundingRect(white)
//img = img[x:x+w, y:y+h]
//
//img = 255 - img
//return img

void get_label_cutted_boundaries(Rect& boundary,int height_before,Mat &cutted){
    int x1 = boundary.x;
    int y1 = boundary.y;
    int x2 = boundary.x + boundary.width;
    int y2 = boundary.y + boundary.height;
    Mat cur_symbol = cutted(Rect(x1,y1-height_before,x2-x1+1,y2-y1+1-height_before));



}

//def get_label_cutted_boundaries(boundary, height_before, cutted):
//# Get the current symbol #
//x1, y1, x2, y2 = boundary
//cur_symbol = cutted[y1-height_before:y2+1-height_before, x1:x2+1]
//
//# Clean and cut #
//cur_symbol = clean_and_cut(cur_symbol)
//cur_symbol = 255 - cur_symbol
//
//# Start prediction of the current symbol #
//feature = extract_hog_features(cur_symbol)
//label = str(model.predict([feature])[0])
//
//return get_target_boundaries(label, cur_symbol, y2)



void detectObjRect(const cv::Mat& src, const cv::Mat& tar, cv::Rect &rect){
    cv::Mat result;
    result.create(src.rows, src.cols, src.type());
    int trackbar_method = TM_CCOEFF_NORMED;
    cv::matchTemplate(src, tar, result,trackbar_method);   //模板匹配
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);       //归一化处理

    //通过minMaxLoc定位最佳匹配位置
    double minValue, maxValue;
    cv::Point minLocation, maxLocation;
    cv::Point matchLocation;
    minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation, cv::Mat());

    //对于方法SQDIFF和SQDIFF_NORMED两种方法来讲，越小的值就有着更高的匹配结果
    //而其余的方法则是数值越大匹配效果越好
    if(trackbar_method == TM_SQDIFF || trackbar_method == TM_SQDIFF_NORMED)
    {
        matchLocation=minLocation;
    }
    else
    {
        matchLocation=maxLocation;
    }

    //TODO zch: rect image need resize
    rect = cv::Rect(matchLocation.x,matchLocation.y,tar.cols,tar.rows);

}

int main(int argc, char **argv) {

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

    Mat image = imread("/home/zch/zuguo1.png",CV_8UC1);

    Mat color = imread("/home/zch/zuguo1.png");



//    Mat image = imread("/home/zch/Orchestra/input/04.PNG",CV_8UC1);
//
//    Mat color = imread("/home/zch/Orchestra/input/04.PNG");

    Mat image_1,image_2;
    fastNlMeansDenoising(image,image_1,10,7,21);
    threshold(image_1,image_2,0,255,THRESH_OTSU);

    int height = image.rows;
    int width = image.cols;


    LOG(INFO) << "height is " << image.rows << " width is " << image.cols;
    imshow("origin", color);

    vector<int> staff_lines;
    vector<int> staff_lines_thicknesses;

    vector<Mat> cutted_images;
    vector<int> cutting_position;

    vector<int> ref_lines;
    vector<int> lines_spacing;

    getStaffLines(width, height, image_2, 0.7, staff_lines, staff_lines_thicknesses);

    LOG(INFO) << "staff lines num is " << staff_lines.size();

    int staff_space = staff_lines[1] - staff_lines[0] + staff_lines_thicknesses[0];

    LOG(INFO) << "line space is " << staff_space;

//
    Mat cleaned = removeStaffLines(image_2, width, staff_lines, staff_lines_thicknesses);
    imshow("cleaned", cleaned);

//
    cutImageIntoBuckets(cleaned, staff_lines,cutting_position,cutted_images);
//
//    // NOTE: cleaned image is image without staff-line
//    getRefLines(cutting_position, staff_lines,ref_lines,lines_spacing);


    string last_acc;
    string last_num;

    int height_before = 0;
    vector<Rect> symbols_boundaries;
    for(int cutted_row = 0;cutted_row < cutted_images.size(); cutted_row++){

        bool is_started = false;
        symbols_boundaries = segmentation(height_before, cutted_images[cutted_row]);
        sort(symbols_boundaries.begin(), symbols_boundaries.end(),
             [](const Rect &a, const Rect &b) {
                 return (a.x < b.x);
             });
//        LOG(INFO) << "cutted height is " << cutted_image.rows << " width is " << cutted_image.cols;


        vector<Rect> merge_symbols_boundaries;
        float y_threshold = 0.3;
        float x_threshold = 0.8;
//        cout << "---------------------" << "symbols_boundaries size is " << symbols_boundaries.size() << "---------------------" << endl;
        for(int i = 0;i < symbols_boundaries.size(); i++) {
            Point2f center_i = Point2f(symbols_boundaries[i].x + symbols_boundaries[i].width / 2, symbols_boundaries[i].y + symbols_boundaries[i].height / 2);

            for(int j = i+1; j < symbols_boundaries.size(); j++) {
                Point2f center_j = Point2f(symbols_boundaries[j].x + symbols_boundaries[j].width / 2, symbols_boundaries[j].y + symbols_boundaries[j].height / 2);
                float merge_distance = abs(center_j.y - center_i.y) * y_threshold + abs(center_j.x - center_i.x) * x_threshold;
                if(merge_distance < 20){
//                    cout << " i height  is " << symbols_boundaries[i].height << " width is " << symbols_boundaries[i].width << endl;
//                    cout << " i x  is " << symbols_boundaries[i].x << " y is " << symbols_boundaries[i].y << endl;
//                    cout << " j height  is " << symbols_boundaries[j].height << " width is " << symbols_boundaries[j].width << endl;
//                    cout << " j x  is " << symbols_boundaries[j].x << " y is " << symbols_boundaries[j].y << endl;

                    symbols_boundaries[i].width = max(symbols_boundaries[j].x + symbols_boundaries[j].width - symbols_boundaries[i].x,symbols_boundaries[i].width);
                    symbols_boundaries[i].height = max(abs(symbols_boundaries[j].y - symbols_boundaries[i].y) + symbols_boundaries[j].height ,symbols_boundaries[i].height);
                    symbols_boundaries[i].x = min(symbols_boundaries[i].x,symbols_boundaries[j].x);
                    symbols_boundaries[i].y = min(symbols_boundaries[i].y,symbols_boundaries[j].y);
//                    cout << "------------ after merge -----------" << endl;

//                    cout << " height  is " << symbols_boundaries[i].height << " width is " << symbols_boundaries[i].width << endl;
//                    cout << " x  is " << symbols_boundaries[i].x << " y is " << symbols_boundaries[i].y << endl;

                }else{
                    merge_symbols_boundaries.push_back(symbols_boundaries[i]);
//                    cout << "------------merge----------" << endl;
//                    cout << "begin i is " << i <<  " end j is " << j << endl;
                    if(j > i + 1){
                        i = j - 1;
                    }
                    break;
                }
            }
            if( i == symbols_boundaries.size() - 1){
                merge_symbols_boundaries.push_back(symbols_boundaries[i]);
            }
        }

//        cout << "------------------------------------------" << endl;



        for(int i = 0; i < merge_symbols_boundaries.size(); i++){

            Rect rectangle = merge_symbols_boundaries[i];

            Mat symbol_image = cutted_images[cutted_row](rectangle);

            Rect symbol_rect;


            rectangle.y = rectangle.y + cutting_position[cutted_row];
            cv::rectangle(color, rectangle, Scalar(0, 0, 255),1, LINE_8,0);




        }

    }

    imshow("result",color);







    waitKey(0);
    return 0;
}

//
//
//for boundary in symbols_boundaries:
//        label, cutted_boundaries = get_label_cutted_boundaries(boundary, height_before, cutted[it])
//
//if label == 'clef':
//is_started = True
//
//for cutted_boundary in cutted_boundaries:
//        _, y1, _, y2 = cutted_boundary
//if is_started == True and label != 'barline' and label != 'clef':
//text = text_operation(label, ref_lines[it], lines_spacing[it], y1, y2)
//
//if (label == 't_2' or label == 't_4') and last_num == '':
//last_num = text
//elif label in accidentals:
//last_acc = text
//else:
//if last_acc != '':
//text = text[0] + last_acc + text[1:]
//last_acc=  ''
//
//if last_num != '':
//text = f'\meter<"{text}/{last_num}">'
//last_num =  ''
//
//not_dot = label != 'dot'
//f.write(not_dot * ' ' + text)
//
//height_before += cutted[it].shape[0]
//f.write(' ]\n')
//
//if len(cutted) > 1:
//f.write('}')

