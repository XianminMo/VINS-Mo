#include "solve_5pts.h"


namespace cv {
    void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
    {

        Mat E = _E.getMat().reshape(1, 3);
        CV_Assert(E.cols == 3 && E.rows == 3);

        Mat D, U, Vt;
        SVD::compute(E, D, U, Vt);

        if (determinant(U) < 0) U *= -1.;
        if (determinant(Vt) < 0) Vt *= -1.;

        Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        W.convertTo(W, E.type());

        Mat R1, R2, t;
        R1 = U * W * Vt;
        R2 = U * W.t() * Vt;
        t = U.col(2) * 1.0;

        R1.copyTo(_R1);
        R2.copyTo(_R2);
        t.copyTo(_t);
    }

    int recoverPose( InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                         OutputArray _R, OutputArray _t, InputOutputArray _mask)
    {

        Mat points1, points2, cameraMatrix;
        _points1.getMat().convertTo(points1, CV_64F);
        _points2.getMat().convertTo(points2, CV_64F);
        _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

        int npoints = points1.checkVector(2);
        CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                                  points1.type() == points2.type());

        CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

        if (points1.channels() > 1)
        {
            points1 = points1.reshape(1, npoints);
            points2 = points2.reshape(1, npoints);
        }

        double fx = cameraMatrix.at<double>(0,0);
        double fy = cameraMatrix.at<double>(1,1);
        double cx = cameraMatrix.at<double>(0,2);
        double cy = cameraMatrix.at<double>(1,2);

        points1.col(0) = (points1.col(0) - cx) / fx;
        points2.col(0) = (points2.col(0) - cx) / fx;
        points1.col(1) = (points1.col(1) - cy) / fy;
        points2.col(1) = (points2.col(1) - cy) / fy;

        points1 = points1.t();
        points2 = points2.t();

        Mat R1, R2, t;
        decomposeEssentialMat(E, R1, R2, t);
        Mat P0 = Mat::eye(3, 4, R1.type());
        Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
        P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
        P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
        P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
        P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

        // Do the cheirality check.
        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between postive and negtive.
        double dist = 50.0;
        Mat Q;
        triangulatePoints(P0, P1, points1, points2, Q);
        Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask1 = (Q.row(2) < dist) & mask1;
        Q = P1 * Q;
        mask1 = (Q.row(2) > 0) & mask1;
        mask1 = (Q.row(2) < dist) & mask1;

        triangulatePoints(P0, P2, points1, points2, Q);
        Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask2 = (Q.row(2) < dist) & mask2;
        Q = P2 * Q;
        mask2 = (Q.row(2) > 0) & mask2;
        mask2 = (Q.row(2) < dist) & mask2;

        triangulatePoints(P0, P3, points1, points2, Q);
        Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask3 = (Q.row(2) < dist) & mask3;
        Q = P3 * Q;
        mask3 = (Q.row(2) > 0) & mask3;
        mask3 = (Q.row(2) < dist) & mask3;

        triangulatePoints(P0, P4, points1, points2, Q);
        Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask4 = (Q.row(2) < dist) & mask4;
        Q = P4 * Q;
        mask4 = (Q.row(2) > 0) & mask4;
        mask4 = (Q.row(2) < dist) & mask4;

        mask1 = mask1.t();
        mask2 = mask2.t();
        mask3 = mask3.t();
        mask4 = mask4.t();

        // If _mask is given, then use it to filter outliers.
        if (!_mask.empty())
        {
            Mat mask = _mask.getMat();
            CV_Assert(mask.size() == mask1.size());
            bitwise_and(mask, mask1, mask1);
            bitwise_and(mask, mask2, mask2);
            bitwise_and(mask, mask3, mask3);
            bitwise_and(mask, mask4, mask4);
        }
        if (_mask.empty() && _mask.needed())
        {
            _mask.create(mask1.size(), CV_8U);
        }

        CV_Assert(_R.needed() && _t.needed());
        _R.create(3, 3, R1.type());
        _t.create(3, 1, t.type());

        int good1 = countNonZero(mask1);
        int good2 = countNonZero(mask2);
        int good3 = countNonZero(mask3);
        int good4 = countNonZero(mask4);

        if (good1 >= good2 && good1 >= good3 && good1 >= good4)
        {
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask1.copyTo(_mask);
            return good1;
        }
        else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
        {
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask2.copyTo(_mask);
            return good2;
        }
        else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
        {
            t = -t;
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask3.copyTo(_mask);
            return good3;
        }
        else
        {
            t = -t;
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask4.copyTo(_mask);
            return good4;
        }
    }

    int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                         OutputArray _t, double focal, Point2d pp, InputOutputArray _mask)
    {
        Mat cameraMatrix = (Mat_<double>(3,3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
        return cv::recoverPose(E, _points1, _points2, cameraMatrix, _R, _t, _mask);
    }
}


/**
 * @brief 求解两帧之间的相对位姿（旋转和平移）
 * 
 * 给定一对两帧间的特征点归一化坐标对应关系（corres），
 * 利用八点法+RANSAC（cv::findFundamentalMat）以及本质矩阵分解（cv::recoverPose），
 * 估算帧间的相对旋转 (Rotation) 和相对平移 (Translation)。
 * 该函数也进行Outlier剔除，最终要求内点数量大于阈值才认为估算有效。
 *
 * 关键步骤解释：
 * 1. 首先判断对应点数量是否足够（至少15对，否则无法鲁棒估计）。
 * 2. 将Eigen点对数据转换为OpenCV的cv::Point2f容器，分别表示左图和右图下的归一化像素坐标。
 * 3. 调用cv::findFundamentalMat计算基础矩阵并剔除异常值（RANSAC容错，阈值0.3/460）。
 * 4. 构建单位内参相机矩阵（本代码中点坐标已经做过归一化）。
 * 5. 利用cv::recoverPose进一步从基础矩阵恢复出“本质矩阵”对应的相对旋转R、平移t，并获取inlier掩码。
 * 6. OpenCV结果转为Eigen格式，注意OpenCV输出为从第一帧到第二帧的变换（R，T）；
 *    本函数输出为第一帧在第二帧坐标系下的“逆变换”，即R^T，-R^T*T。
 * 7. 最后检查inlier数量是否足够，大于12则判定估计成功，返回true，否则false。
 *
 * @param corres    输入，对应的特征点对（每对包含两帧的归一化归一像素坐标）
 * @param Rotation  输出，相对旋转R（从后帧到前帧，3x3矩阵）
 * @param Translation 输出，相对平移t（从后帧到前帧，3x1向量）
 * @return bool     若估计成功返回true，否则false
 */
bool MotionEstimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    // 对应点数量不足，直接失败
    if (corres.size() < 15)
        return false;

    // 1. 将输入Eigen点对转为OpenCV的cv::Point2f
    vector<cv::Point2f> pts1, pts2;
    for (size_t i = 0; i < corres.size(); i++)
    {
        pts1.push_back(cv::Point2f(static_cast<float>(corres[i].first(0)), static_cast<float>(corres[i].first(1))));
        pts2.push_back(cv::Point2f(static_cast<float>(corres[i].second(0)), static_cast<float>(corres[i].second(1))));
    }

    // 2. RANSAC 求基础矩阵, 并利用匹配掩码剔除离群点
    cv::Mat mask;
    // RANSAC内点阈值是像素单位，这里点已经被单位化，所以阈值极小
    cv::Mat E = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);

    // 3. 假设单位内参（归一化像素）
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                                       0, 1, 0,
                                                       0, 0, 1);

    // 4. 利用基础矩阵恢复相对R,t（本质矩阵分解）
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, pts1, pts2, cameraMatrix, rot, trans, mask);
    // inlier_cnt: 恢复出的内点数量

    // 5. OpenCV矩阵->Eigen转换，并变换为“从后帧到前帧”的逆变换
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    for (int i = 0; i < 3; i++)
    {
        t_eigen(i) = trans.at<double>(i, 0);
        for (int j = 0; j < 3; j++)
            R_eigen(i, j) = rot.at<double>(i, j);
    }

    // 6. OpenCV输出为"从前到后"(R, t)。输出需转为W系下前->后变换的逆: R^T, -R^T*t
    Rotation = R_eigen.transpose();
    Translation = -R_eigen.transpose() * t_eigen;

    // 7. 检查内点数量是否达到阈值要求
    if (inlier_cnt > 12)
        return true;
    else
        return false;
}



