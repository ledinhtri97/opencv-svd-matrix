#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

cv::Mat find_T_matrix(cv::Mat pts, cv::Mat t_ptsh) 
{
    // cv::Mat H;
    //PTS: (3x4)
    //T_PTS: (3x4)
    cv::Mat A = cv::Mat::zeros(8,9, CV_32F);
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned k = 3,j = 0; k < 6; k++, j++)
        {
            A.at<float>(i*2, k) = -t_ptsh.at<float>(2,i)*pts.at<float>(j, i);
        }

        for (unsigned k = 6,j = 0; k < 9; k++, j++)
        {
            A.at<float>(i*2, k) = t_ptsh.at<float>(1,i)*pts.at<float>(j, i);
        }

        for (unsigned k = 0,j = 0; k < 3; k++, j++)
        {
            A.at<float>(i*2+1, k) = t_ptsh.at<float>(2,i)*pts.at<float>(j, i);
        }

        for (unsigned k = 6,j = 0; k < 9; k++, j++)
        {
            A.at<float>(i*2+1, k) = -t_ptsh.at<float>(0,i)*pts.at<float>(j, i);
        }
    }

    // cout << "pts = \n " << pts << "\n\n";
    // cout << "t_ptsh = \n " << t_ptsh << "\n\n";
    // cout << "A = \n " << A << "\n\n";

    cv::SVD doSVD(A, cv::SVD::FULL_UV);
    doSVD(A, cv::SVD::FULL_UV);

    // cout << "u = \n " << doSVD.u << "\n\n";
    // cout << "vt = \n " << doSVD.vt << "\n\n";
    // cout << "w = \n " << doSVD.w << "\n\n";
    
    return doSVD.vt.row(8);
}
int main( int argc, char** argv )
{
	float net_stride = 16;
	float side = ((208. + 40.)/2)/net_stride;

	// float base_arr[3][4] = {{-0.5, 0.5, 0.5, -0.5}, {-0.5, -0.5, 0.5, 0.5}, {1.0, 1.0, 1.0, 1.0}};
	float affine_arr[2][3];

	float tmp[6] = { 0.61783755, -0.01091473,  0.02857949,
				 -0.10235066,  0.19068556,  0.05085495};

	// float MN_arr[2][4] = {{18.0, 18.0, 18.0, 18.0}, {18.0, 18.0, 18.0, 18.0}};
	// Mat MN(2,4, CV_32F, MN_arr);
	Mat MN = (cv::Mat_<float>(2,4) << 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0);

	unsigned int c = 0;
	for (unsigned int i = 0; i < 2; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			if ((i == 0 && j == 0) || (i == 1 && j == 1))
				affine_arr[i][j] = MAX(tmp[c], 0.0);
			else
				affine_arr[i][j] = tmp[c];
			c++;
		}
	}
	// float mn_array[2][4] = {{9+0.5, 9+0.5, 9+0.5, 9+0.5}, {14+0.5, 14+0.5, 14+0.5, 14+0.5}};
	// Mat mn(2,4, CV_32F, mn_array);
    Mat affine(2,3, CV_32F, affine_arr);
    // Mat base(3,4, CV_32F, base_arr);

    float I = 9.5;
    float J = 16.5;
    Mat mn = (cv::Mat_<float>(2,4) << I, I, I, I, J, J, J, J);
    Mat base = (cv::Mat_<float>(3,4) << -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0);

    Mat pts = affine * base;
    Mat pts_MN_center_mn = pts*side;
    Mat pts_MN = pts_MN_center_mn + mn;
    Mat pts_prop = pts_MN / MN;

    // cout << "affine = \n " << affine << "\n\n";
    // cout << "base = \n " << base << "\n\n";
    // cout << "pts = \n " << pts << "\n\n";
    // cout << "pts_MN_center_mn = \n " << pts_MN_center_mn << "\n\n";
    // cout << "pts_MN = \n " << pts_MN << "\n\n";
    cout << "pts_prop = \n " << pts_prop << "\n\n";

    cv::Mat t_ptsh = (cv::Mat_<float>(3, 4) << 0.0, 240.0, 240.0, 0.0, 0.0, 0.0, 80.0, 80.0, 1.0, 1.0, 1.0, 1.0);

    float netw = 288.0;
    float neth = 288.0;
    cv::Mat wh = (cv::Mat_<float>(2,2) << netw, 0, 0, neth);

    cv::Mat cols = (cv::Mat_<float>(1,4) << 1.0, 1.0, 1.0, 1.0);

    // cout << "wh = \n " << wh << "\n\n";
    cv::Mat ptsh = pts_prop.t() * wh;
    // cout << "ptsh = \n " << ptsh << "\n\n";
    // cout << "cols = \n " << cols << "\n\n";
    cv::Mat out_ptsh;

    cv::vconcat(ptsh.t(), cols, out_ptsh);
    // cout << "out_ptsh = \n " << out_ptsh << "\n\n";

    Mat H =  find_T_matrix(out_ptsh, t_ptsh);

    cout << "H = \n " << H << "\n\n";

    // cout << "rows = \n " << H.rows << "\n\n";
    // cout << "cols = \n " << H.cols << "\n\n";
    // for (unsigned int i = 0; i < H.cols; i++)
    // {
    //     cout << H.at<float>(0, i) << ',';
    // }

    return 0;
}