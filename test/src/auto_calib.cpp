#include "auto_calib.h"

bool AutoCalib::getPointcloud(std::string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr ptcloud) //�����ļ��У��õ�����
{
    // Load the actual pointcloud.
    const size_t kMaxNumberOfPoints = 1e6; // From Readme for raw files.
    ptcloud->clear();
    ptcloud->reserve(kMaxNumberOfPoints);
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    if (!input)
    {
        std::cout << "Could not open pointcloud file.\n";
        return false;
    }

    // From yanii's kitti-pcl toolkit:
    // https://github.com/yanii/kitti-pcl/blob/master/src/kitti2pcd.cpp
    for (size_t i = 0; input.good() && !input.eof(); i++)
    {
        pcl::PointXYZI point;
        input.read((char *)&point.x, 3 * sizeof(float));
        input.read((char *)&point.intensity, sizeof(float));
        ptcloud->push_back(point);
    }
    input.close();
    return true;
}
void AutoCalib::getOxtsData(std::string filename, std::vector<float> &oxts_vec) //oxts_vec��gps�źţ����Ƶ���֡�ϳ�һ֡
{
    std::ifstream oxtsfile(filename);
    std::string line_data;
    if (oxtsfile)
    {
        while (getline(oxtsfile, line_data))
        {
            // std::cout<<"line_data = ";
            // for(int i = 0; i < line_data.size(); i++)
            // {
            //     std::cout<<line_data[i]<<" ";
            // }
            // std::cout<<std::endl;
            //stringתchar
            char *lineCharArray;
            const int len = line_data.length();
            lineCharArray = new char[len + 1];
            strcpy(lineCharArray, line_data.c_str());

            char *p;                        //�ָ�����ַ���??
            p = strtok(lineCharArray, " "); //����spaceChar�ָ�
            //�����ݼ���vector��
            while (p)
            {
                // std::cout<<atof(p)<<" ";
                oxts_vec.push_back(atof(p));
                p = strtok(NULL, " ");
            }
        }
    }
    else
    {
        std::cout << "Can not open oxts file" << std::endl;
    }
}

bool AutoCalib::point_cmp(pcl::PointXYZI a, pcl::PointXYZI b) //д��һ���?
{
    return a.z < b.z;
}

void AutoCalib::icp_erase(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc1, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc2, Eigen::Matrix4f &T_v, pcl::PointCloud<pcl::PointXYZI>::Ptr &dynamic_cloud)
{

    float max_range; //0.05���кܶ�Ŀ��,0.005�Ѿ���Ч���ˣ������Ż�������ƫ�ˣ�û�޳��ɾ�
    //֮ǰ�ľ��鲻��Ч
    //���õ���ֵ  2̫��0.002̫С,0.01 little,next:(2+0.01)/2=1,1���Ǵ��ˣ���δ�޳�����1+0.01��/2=0.5 still big ;  try 0.1 already ;  ��С��0.05��ֻ�ж�Ŀ��
    // Transform the input dataset using the final transformation
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    //pcl::transformPointCloud(*pc2, *output_cloud, T_v.inverse());
    pcl::transformPointCloud(*pc2, *output_cloud, T_v);
    pcl::KdTreeFLANN<pcl::PointXYZI> tree_;
    tree_.setInputCloud(output_cloud); //����kdtree��������pc1��ÿ����

    // //���ӻ����뿪ʼ
    //     //������ʼ��Ŀ��
    //      pcl::visualization::PCLVisualizer viewer("registration Viewer");

    //      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> src_h(pc1, 0, 255, 0);  //��ɫ GREEN

    //      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> twice_h(pc2, 0, 0, 255); //��ɫ BLUE

    //      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> tgt_h(output_cloud, 255, 0, 0);  //��ɫ RED

    //      viewer.setBackgroundColor(0,0,0);
    //      viewer.addPointCloud(pc1, src_h, "source cloud");
    //      viewer.addPointCloud(pc2, twice_h, "twice h");
    //      viewer.addPointCloud(output_cloud, tgt_h, "tgt cloud");

    //      while (!viewer.wasStopped())
    //     {
    //          viewer.spinOnce(100);
    //          boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    //      }
    // //���ӻ��������?

    int nearK = 1; //����ٵ����

    // For each point in the source dataset

    std::vector<int> del;
    std::vector<int> sum;

    // //�ⲿ��Ϊ�˿��ӻ�����
    // std::vector<int> range1;
    // std::vector<int> range2;
    // std::vector<int> range3;
    // std::vector<int> range4;
    // pcl::PointCloud<pcl::PointXYZI>::Ptr range1_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr range2_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr range3_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr range4_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    for (size_t i = 0; i < pc1->points.size(); ++i) //����pc1�ĵ㣬�ڶ���
    {

        std::vector<int> nn_indices(nearK);
        std::vector<float> nn_dists(nearK);
        // Find its nearest neighbor in the target
        float dist = std::sqrt(std::pow(pc1->points[i].y, 2) + std::pow(pc1->points[i].x, 2) + std::pow(pc1->points[i].z, 2));
        tree_.nearestKSearch(pc1->points[i], nearK, nn_indices, nn_dists); //��ѯ�����?���룬�ڽ�����������������ڽ���ľ���ƽ��ֵ
        //std::cout << "nn_indices[0]= " << nn_indices[0] << std::endl;
        //std::cout << "for this i = " << i << ",dist = " << dist << std::endl;
        max_range = 0.01 * dist;
        //max_range = 0.008*(dist-7);
        //max_range =  0.005+0.01*(dist-7);
        //max_range = 0.005+0.00025*(dist-7);

        //    if(dist<=7)
        //     {
        //         max_range = 0.005;
        //     }
        //     else if(dist>7&&dist<=15)
        //     {
        //         max_range = 0.005+0.0035*(dist-7);   //�ο���ԭ���Ĺ�ʽ0.005+0.00025*()
        //     }
        //     else if(dist>15&&dist<=25)
        //     {
        //        max_range = 0.005+0.0035*8+0.020*(dist-15);
        //     }
        //     else if(dist>25)
        //     {
        //         max_range = 0.005+0.0035*8+0.020*10+0.2*(dist-25);
        //     }

        // //д��max_range�ֲ�Ŀ��ӻ������ڵ��Ƶ㲻��XYZRGB�ģ����Բ���ֱ��������ɫ���ֳ��Ķε��ƣ���һ��
        // if(dist<=7)
        // {
        //     range1.push_back(i);
        // }
        // else if(dist>7&&dist<=15)
        // {
        //     range2.push_back(i);
        // }
        // else if(dist>15&&dist<=25)
        // {
        //     range3.push_back(i);
        // }
        // else if(dist>25)
        // {
        //     range4.push_back(i);
        // }

        //Դ���룬���ӻ�֮ǰ����Դ����
        if (nn_dists[0] >= max_range)
        {
            del.push_back(i);
        }
        else
        {
            sum.push_back(i);
        }
    }

    pcl::copyPointCloud(*pc1, del, *dynamic_cloud); //cloud_in  index  cloud_out ,only choosen index to output
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*pc1, sum, *tmp_cloud);
    *pc1 = *tmp_cloud;
    //pc1=tmp_cloud;

    //         pcl::copyPointCloud(*pc1, range1, *range1_cloud);
    //         pcl::copyPointCloud(*pc1, range2, *range2_cloud);
    //         pcl::copyPointCloud(*pc1, range3, *range3_cloud);
    //         pcl::copyPointCloud(*pc1, range4, *range4_cloud);

    //     //���ӻ�����
    //     //������ʼ��Ŀ��
    //      pcl::visualization::PCLVisualizer viewer("registration Viewer");

    //      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> show1(range1_cloud, 0, 0, 255); //��ɫ BLUE

    //      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> show2(range2_cloud, 0, 255, 0);  //��ɫ GREEN

    //      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> show3(range3_cloud, 255, 0, 0);  //��ɫ RED

    //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> show4(range4_cloud, 255, 255,0); //yellow

    //      viewer.setBackgroundColor(0,0,0);
    //      viewer.addPointCloud(range1_cloud, show1, "range1");
    //      viewer.addPointCloud(range2_cloud, show2, "range2");
    //      viewer.addPointCloud(range3_cloud, show3, "range3");
    //       viewer.addPointCloud(range4_cloud, show4, "range4");

    //      while (!viewer.wasStopped())
    //     {
    //          viewer.spinOnce(100);
    //          boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    //      }
    // //���ӻ��������?
}


void AutoCalib::evaluation_function(pcl::PointCloud<pcl::PointXYZI>::Ptr &infer_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &filter_cloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr &truth_cloud,std::vector <float> &score_count_cloud)
{
    
//���ӻ�����ĵ���
    // pcl::visualization::PCLVisualizer viewer("registration Viewer");

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> infer_h(infer_cloud, 0, 255, 0);  // GREEN

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> filter_h(filter_cloud, 0, 0, 255); // BLUE

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> tgt_h(truth_cloud, 255, 0, 0);  // RED

    // viewer.setBackgroundColor(0,0,0);
    // viewer.addPointCloud(infer_cloud, infer_h, "infer_cloud");
    // viewer.addPointCloud(filter_cloud, filter_h, "filter_cloud");
    // viewer.addPointCloud(truth_cloud, tgt_h, "truth_cloud");

    // while (!viewer.wasStopped()){
    //     viewer.spinOnce(100);
    //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }

//����ƽ��ŷʽ��������ָ��
#define Dists_evaluation
#ifdef Dists_evaluation
    pcl::KdTreeFLANN<pcl::PointXYZI> tree_;

    tree_.setInputCloud(truth_cloud); 

    int nearK = 1; 

    // For each point in the infer cloud
    for (size_t i = 0; i < infer_cloud->points.size(); ++i) 
    {
        std::vector<int> nn_indices(nearK);
        std::vector<float> nn_dists(nearK);

        // Find its nearest neighbor in the target
        tree_.nearestKSearch(infer_cloud->points[i], nearK, nn_indices, nn_dists); 

        //std::cout << "infer_cloud  nn_indices[0]= " << nn_indices[0] << std::endl;
        score_count_cloud[0] +=nn_dists[0];
    }
    score_count_cloud[0] = score_count_cloud[0]/infer_cloud->points.size();

    for (size_t i = 0; i < filter_cloud->points.size(); ++i) 
    {
        std::vector<int> nn_indices(nearK);
        std::vector<float> nn_dists(nearK);

        // Find its nearest neighbor in the target
        tree_.nearestKSearch(filter_cloud->points[i], nearK, nn_indices, nn_dists); 

        //std::cout << "filter_cloud nn_indices[0]= " << nn_indices[0] << std::endl;
        score_count_cloud[1]+=nn_dists[0];
    }
    score_count_cloud[1] = score_count_cloud[1]/filter_cloud->points.size();
#endif

#define NDT_evaluation
#ifdef NDT_evaluation
    double TransformationEpsilon = config["TransformationEpsilon"].as<double>();
    double StepSize = config["StepSize"].as<double>();
    float Resolution = config["Resolution"].as<float>();
    int NDT_MaximumIterations = config["NDT_MaximumIterations"].as<int>();
    float LeafSize = config["LeafSize"].as<float>();

    //infer_cloud NDT
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_1 (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter_1;
    voxel_filter_1.setLeafSize (LeafSize, LeafSize, LeafSize);
    voxel_filter_1.setInputCloud (infer_cloud);
    voxel_filter_1.filter (*filtered_cloud_1);
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt_1;

    ndt_1.setTransformationEpsilon (TransformationEpsilon);
    //ΪMore-Thuente������������󲽳�
    ndt_1.setStepSize (StepSize);
    //����NDT����ṹ�ķֱ��ʣ�VoxelGridCovariance��
    ndt_1.setResolution (Resolution);
    //����ƥ�������������
    ndt_1.setMaximumIterations (NDT_MaximumIterations);
    // ����Ҫ��׼�ĵ���
    ndt_1.setInputSource (filtered_cloud_1);
    //���õ�����׼Ŀ��
    ndt_1.setInputTarget (truth_cloud);
    //����ndt����
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out_1(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f init_guess_1 = Eigen::Matrix4f::Identity();
    ndt_1.align(*cloud_out_1,init_guess_1);
    //ʹ�ô����ı任��δ���˵�������ƽ��б任
    pcl::transformPointCloud (*infer_cloud, *cloud_out_1, ndt_1.getFinalTransformation ());

    //filter_cloud NDT
    //�������ɨ�����,���ƥ����ٶȡ�
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setLeafSize (LeafSize, LeafSize, LeafSize);
    voxel_filter.setInputCloud (filter_cloud);
    voxel_filter.filter (*filtered_cloud);
    //��ʼ����̬�ֲ��任��NDT��
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    //���������߶�NDT����
    //Ϊ��ֹ����������Сת������
    ndt.setTransformationEpsilon (TransformationEpsilon);
    //ΪMore-Thuente������������󲽳�
    ndt.setStepSize (StepSize);
    //����NDT����ṹ�ķֱ��ʣ�VoxelGridCovariance��
    ndt.setResolution (Resolution);
    //����ƥ�������������
    ndt.setMaximumIterations (NDT_MaximumIterations);
    // ����Ҫ��׼�ĵ���
    ndt.setInputSource (filtered_cloud);
    //���õ�����׼Ŀ��
    ndt.setInputTarget (truth_cloud);
    //����ndt����
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    ndt.align(*cloud_out,init_guess);
    //ʹ�ô����ı任��δ���˵�������ƽ��б任
    pcl::transformPointCloud (*filter_cloud, *cloud_out, ndt.getFinalTransformation ());

    //�ñ任������Ϊ���۷���
    score_count_cloud[2] = ndt_1.getFinalNumIteration();
    score_count_cloud[3] = ndt.getFinalNumIteration();

    //�Ƿ���NDT�������ӻ�
    if (config["NDT_visualization"].as<bool>()){
        // ��ʼ�����ƿ��ӻ�����
        boost::shared_ptr<pcl::visualization::PCLVisualizer>
        viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer_final->setBackgroundColor (0, 0, 0);
        //��truth������ɫ����ɫ�������ӻ�
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        target_color (truth_cloud, 255, 0, 0);
        viewer_final->addPointCloud<pcl::PointXYZI> (truth_cloud, target_color, "target cloud");
        viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "target cloud");
        //��ת�����infer������ɫ����ɫ�������ӻ�
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        output_color_1 (cloud_out_1, 0, 255, 0);
        viewer_final->addPointCloud<pcl::PointXYZI> (cloud_out_1, output_color_1, "output infer_cloud");
        viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "output infer_cloud");
        //��ת�����filter������ɫ����ɫ�������ӻ�
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        output_color (cloud_out, 0, 0, 255);
        viewer_final->addPointCloud<pcl::PointXYZI> (cloud_out, output_color, "output filter_cloud");
        viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "output filter_cloud");
        // �������ӻ�
        viewer_final->addCoordinateSystem (1.0);
        viewer_final->initCameraParameters ();
        //�ȴ�ֱ�����ӻ����ڹرա�
        while (!viewer_final->wasStopped ())
        {
            viewer_final->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }  
    }

    
#endif

//����ICP����ָ��
#define ICP_evaluation
#ifdef ICP_evaluation

    //��config�ļ��ж�icp����
    double MaxCorrespondenceDistance = config["MaxCorrespondenceDistance"].as<double>();
    double ICP_TransformationEpsilon = config["ICP_TransformationEpsilon"].as<double>();
    double EuclideanFitnessEpsilon = config["EuclideanFitnessEpsilon"].as<double>();
    int ICP_MaximumIterations = config["ICP_MaximumIterations"].as<int>();
    
    //infer_cloud ICP
    //����ICP��������ICP��׼
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp_1;
    
    icp_1.setInputSource(infer_cloud); 
    icp_1.setInputTarget(truth_cloud); 
    
    icp_1.setMaxCorrespondenceDistance (MaxCorrespondenceDistance);
    icp_1.setMaximumIterations (ICP_MaximumIterations);
    icp_1.setTransformationEpsilon (ICP_TransformationEpsilon);
    icp_1.setEuclideanFitnessEpsilon (EuclideanFitnessEpsilon);
    //�洢���
    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_out_1(new pcl::PointCloud<pcl::PointXYZI>); 

    // pcl::PointCloud<pcl::PointXYZI>::Ptr temp_1(new pcl::PointCloud<pcl::PointXYZI>); 

    //������׼
    // score_count_cloud[4] = 0;
    // int interations_1 = 1;
    // for(int i = 0; i < 50; i++){
    //     icp_1.align(*icp_out_1);
    //     interations_1++;
    //     icp_1.setMaximumIterations(interations_1);
    //     score_count_cloud[4]++;
    //     if(icp_1.getTransformationEpsilon() <= ICP_TransformationEpsilon) break;
    // }

    icp_1.align(*icp_out_1);
    score_count_cloud[4] = icp_1.getFitnessScore();

    //ʹ�ô����ı任��δ���˵�������ƽ��б任
    pcl::transformPointCloud (*infer_cloud, *icp_out_1, icp_1.getFinalTransformation ());


    //filter_cloud ICP
    //����ICP��������ICP��׼
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    
    icp.setInputSource(filter_cloud); 
    icp.setInputTarget(truth_cloud); 
    
    icp.setMaxCorrespondenceDistance (MaxCorrespondenceDistance);
    icp.setMaximumIterations (ICP_MaximumIterations);
    icp.setTransformationEpsilon (ICP_TransformationEpsilon);
    icp.setEuclideanFitnessEpsilon (EuclideanFitnessEpsilon);
    //�洢���
    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_out(new pcl::PointCloud<pcl::PointXYZI>); 
    //������׼
    // score_count_cloud[5] = 0;
    // bool converged_2 = false;
    // int interations = 1;
    // for(int i = 0; i < 50; i++){
    //     icp.align(*icp_out);
    //     interations++;
    //     icp.setMaximumIterations(interations);
    //     score_count_cloud[5]++;
    //     if(icp.getTransformationEpsilon() <= TransformationEpsilon) break;
    // }

    icp.align(*icp_out);
    score_count_cloud[5] = icp.getFitnessScore();



    //ʹ�ô����ı任��δ���˵�������ƽ��б任
    pcl::transformPointCloud (*filter_cloud, *icp_out, icp.getFinalTransformation ());

    //�Ƿ���ICP���ӻ�
    if (config["ICP_visualization"].as<bool>()){
        // ��ʼ�����ƿ��ӻ�����
        boost::shared_ptr<pcl::visualization::PCLVisualizer>
        viewer_final_icp (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer_final_icp->setBackgroundColor (0, 0, 0);
        //��truth������ɫ����ɫ�������ӻ�
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        target_color_icp (truth_cloud, 255, 0, 0);
        viewer_final_icp->addPointCloud<pcl::PointXYZI> (truth_cloud, target_color_icp, "target cloud");
        viewer_final_icp->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "target cloud");
        //��ת�����infer������ɫ����ɫ�������ӻ�
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        output_color_1_icp (icp_out_1, 0, 255, 0);
        viewer_final_icp->addPointCloud<pcl::PointXYZI> (icp_out_1, output_color_1_icp, "output infer_cloud");
        viewer_final_icp->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "output infer_cloud");
        //��ת�����filter������ɫ����ɫ�������ӻ�
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        output_color_icp (icp_out, 0, 0, 255);
        viewer_final_icp->addPointCloud<pcl::PointXYZI> (icp_out, output_color_icp, "output filter_cloud");
        viewer_final_icp->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "output filter_cloud");
        // �������ӻ�
        viewer_final_icp->addCoordinateSystem (1.0);
        viewer_final_icp->initCameraParameters ();
        //�ȴ�ֱ�����ӻ����ڹرա�
        while (!viewer_final_icp->wasStopped ())
        {
            viewer_final_icp->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }  
    }

#endif

        
}

void AutoCalib::point_cb(pcl::PointCloud<pcl::PointXYZI>::Ptr data, pcl::PointCloud<pcl::PointXYZI>::Ptr final_no_ground)
//ȥ�����Ƶ����µĵ�
{
    // 1.Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // For mark ground points and hold all points
    pcl::PointCloud<pcl::PointXYZI>::Ptr data_org(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*data, *data_org);

    float angle;
    PointXYZIRL point;
    pcl::PointCloud<PointXYZIRL>::Ptr g_all_pc(new pcl::PointCloud<PointXYZIRL>);

    for (size_t i = 0; i < data->points.size(); i++)
    {
        point.x = data->points[i].x;
        point.y = data->points[i].y;
        point.z = data->points[i].z;
        point.intensity = data->points[i].intensity;

        point.label = 0u; // 0 means uncluster
        g_all_pc->points.push_back(point);
    }
    //std::vector<int> indices;
    //pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn,indices);
    // 2.Sort on Z-axis value.
    sort(data_org->points.begin(), data_org->points.end(), AutoCalib::point_cmp);
    // 3.Error point removal
    // As there are some error mirror reflection under the ground,
    // here regardless point under 2* sensor_height
    // Sort point according to height, here uses z-axis in default
    // �����˴������߶�
    pcl::PointCloud<pcl::PointXYZI>::iterator it = data_org->points.begin();
    for (int i = 0; i < data_org->points.size(); i++)
    {
        if (data_org->points[i].z < -1.5 * 2.0)
        {
            it++;
        }
        else
        {
            break;
        }
    }
    data_org->erase(data_org->points.begin(), it);
    // 4. Extract init ground seeds.
    double sum = 0;
    int cnt = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_seeds_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // Calculate the mean height value.
    for (int i = 0; i < data_org->points.size() && cnt < 20; i++)
    {
        sum += data_org->points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0; // in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < data_org->points.size(); i++)
    {
        if (data_org->points[i].z < lpr_height + 0.4)
        {
            g_seeds_pc->points.push_back(data_org->points[i]);
        }
    }

    g_ground_pc = g_seeds_pc;
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_not_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // 5. Ground plane fitter mainloop
    float d_, th_dist_d_;
    Eigen::MatrixXf normal_;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Matrix3f cov;
        Eigen::Vector4f pc_mean;
        pcl::computeMeanAndCovarianceMatrix(*g_ground_pc, cov, pc_mean);
        // Singular Value Decomposition: SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
        // use the least singular vector as normal
        normal_ = (svd.matrixU().col(2));
        // mean ground seeds value
        Eigen::Vector3f seeds_mean = pc_mean.head<3>();

        // according to normal.T*[x,y,z] = -d
        d_ = -(normal_.transpose() * seeds_mean)(0, 0);
        // set distance threhold to `th_dist - d`
        th_dist_d_ = 0.3 - d_;

        g_ground_pc->clear();
        g_not_ground_pc->clear();

        //pointcloud to matrix
        Eigen::MatrixXf points(data->points.size(), 3);
        int j = 0;
        for (auto p : data->points)
        {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int r = 0; r < result.rows(); r++)
        {
            if (result[r] < th_dist_d_)
            {
                g_all_pc->points[r].label = 1u; // means ground
                g_ground_pc->points.push_back(data->points[r]);
            }
            else
            {
                g_all_pc->points[r].label = 0u; // means not ground and non clusterred
                g_not_ground_pc->points.push_back(data->points[r]);
            }
        }
    }

    pcl::copyPointCloud(*g_not_ground_pc, *final_no_ground); //������û�е����ϵĵ�

    // ROS_INFO_STREAM("origin: "<<g_not_ground_pc->points.size()<<" post_process: "<<final_no_ground->points.size());

    // publish ground points
    //    sensor_msgs::PointCloud2 ground_msg;
    //    pcl::toROSMsg(*g_ground_pc, ground_msg);
    //    ground_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    ground_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_ground_.publish(ground_msg);
    //
    //    // publish not ground points
    //    sensor_msgs::PointCloud2 groundless_msg;
    //    pcl::toROSMsg(*final_no_ground, groundless_msg);
    //    groundless_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    groundless_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_no_ground_.publish(groundless_msg);
    //
    //    // publish all points
    //    sensor_msgs::PointCloud2 all_points_msg;
    //    pcl::toROSMsg(*g_all_pc, all_points_msg);
    //    all_points_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    all_points_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_all_points_.publish(all_points_msg);
    //std::cout << g_ground_pc->size() << std::endl;
}

//读语义图片
void AutoCalib::Get_other_images(std::string txtName, std::string folderName, std::vector<cv::Mat> &other_images) 
{
    std::cout << "Start Read Other Images ..." << std::endl;
    std::string filename;
    std::ifstream readtxt;
    readtxt.open(txtName);
    if(readtxt.is_open())
    {
        std::cout << "Get_message_filter_cloud succeed! " << std::endl;
    }
    if (!readtxt)
    {
        std::cout << "\033[31mGet_message_filter_cloud Error: Open txt file faile!\033[0m" << std::endl;
        std::exit(0);
    }

    while (readtxt >> filename)
    {
        filename = folderName + filename;
        cv::Mat other_image = cv::imread(filename);
        other_images.push_back(other_image);
    }
    readtxt.close();
}

//读插值法点云
void AutoCalib::Get_message_filter_cloud(std::string txtName, std::string folderName, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &message_filter_cloud)  //��ȡ�µ�list
{
    std::cout << "Start Read Message_filter Data ..." << std::endl;
    std::string filename;
    std::ifstream readtxt;
    readtxt.open(txtName);
    if(readtxt.is_open())
    {
        std::cout << "Get_message_filter_cloud succeed! " << std::endl;
    }

    if (!readtxt)
    {
        std::cout << "\033[31mGet_message_filter_cloud Error: Open txt file faile!\033[0m" << std::endl;
        std::exit(0);
    }

    while (readtxt >> filename)
    {
        filename = folderName + filename;

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>);
        if (!getPointcloud(filename, raw))  
        {
            std::cout << "\033[31mGet_message_filter_cloud Error: Could not open " << filename << " !\033[0m" << std::endl;
            std::exit(0);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        if (config["delete_ground"].as<bool>())  
        {
            point_cb(raw, pc);
        }
        else
        {
            pcl::copyPointCloud(*raw, *pc);
        }
        message_filter_cloud.push_back(pc);
    }
    readtxt.close();
}

void AutoCalib::Get_truth_cloud(std::string txtName, std::string folderName, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &truth_cloud)  //��ȡ�µ�list
{
    std::cout << "Start Reading truth_cloud Data ..." << std::endl;
    std::string filename;
    std::ifstream readtxt;
    readtxt.open(txtName);
    if(readtxt.is_open())
    {
        std::cout << "Get_truth_cloud succeed! " << std::endl;
    }

    if (!readtxt)
    {
        std::cout << "\033[31mGet_truth_cloud Error: Open txt file faile!\033[0m" << std::endl;
        std::exit(0);
    }

    //int n = 0;

    while (readtxt >> filename)
    {
        filename = folderName + filename;

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>);
        if (!getPointcloud(filename, raw))  
        {
            std::cout << "\033[31mGet_truth_cloud Error: Could not open " << filename << " !\033[0m" << std::endl;
            std::exit(0);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        if (config["delete_ground"].as<bool>())  
        {
            point_cb(raw, pc);
        }
        else
        {
            pcl::copyPointCloud(*raw, *pc);
        }
        truth_cloud.push_back(pc);
    }
    readtxt.close();
}

int AutoCalib::getData(std::string txtName, std::string folderName, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pointclouds,
                       std::vector<cv::Mat> &images, std::vector<float> &oxts_vec)
{
    std::cout << "Start Read Data ..." << std::endl;
    std::string filename;
    std::ifstream readtxt;
    readtxt.open(txtName);
    if (!readtxt)
    {
        std::cout << "\033[31mgetData Error: Open txt file faile!\033[0m" << std::endl;
        std::exit(0);
    }
    int n = 0;
    while (readtxt >> filename)
    {
        filename = folderName + filename;

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>);
        if (!getPointcloud(filename, raw)) //�������е��Ƶĵ�
        {
            std::cout << "\033[31mgetData Error: Could not open " << filename << " !\033[0m" << std::endl;
            std::exit(0);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        if (config["delete_ground"].as<bool>()) //ɾ�������µĵ�
        {
            point_cb(raw, pc);
        }
        else
        {
            pcl::copyPointCloud(*raw, *pc);
        }
        pointclouds.push_back(pc);

        if (!(readtxt >> filename))
        {
            std::cout << "\033[31mgetData Error: no image!\033[0m" << std::endl;
            std::exit(0);
        }
        std::string result_file_temp = folderName + "result/" + filename;
        filename = folderName + filename;
        result_file.push_back(result_file_temp);
        cv::Mat image = cv::imread(filename);
        images.push_back(image);

        if (!(readtxt >> filename))
        {
            std::cout << "\033[31oxtsData Error: no oxts!\033[0m" << std::endl;
            std::exit(0);
        }
        else
        {
            filename = folderName + filename;
            // std::cout<<"finename = "<<filename<<std::endl;
            getOxtsData(filename, oxts_vec);
        }
        n++;
    }
    // std::cout<<"oxts_vec size = " << oxts_vec.size() << std::endl;
    // for(int i = 0; i < oxts_vec.size(); i++)
    // {
    //     std::cout << "oxts = " << oxts_vec[i] << std::endl;
    // }
    // std::cout<<"images.size = "<<images.size()<<" " << "pc.size = " << pointclouds.size() << std::endl;
    std::cout << "Finish Read Data." << std::endl;
    return n;
}

//������ƥ���??�Ӿ��취�õ�ͼ����R T��ICP�õ��״��ı任���������������۱궨��˼·����Ϊ���Ȳ��þ�������
void AutoCalib::find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                                     std::vector<cv::KeyPoint> &keypoints_1,
                                     std::vector<cv::KeyPoint> &keypoints_2,
                                     std::vector<cv::DMatch> &matches)
{

    //-- ��ʼ��
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- ��һ��:���?? Oriented FAST �ǵ�λ��
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // #define ROI_LEFT 0
    // #define ROI_TOP images[ppp].rows/4
    // #define ROI_WIDTH images[ppp].cols
    // #define ROI_HEIGHT images[ppp].rows/2

    // std::vector<cv::KeyPoint> keypoints_1_temp, keypoints_2_temp;

    // for(int i = 0; i < keypoints_1.size(); ++i)
    // {
    //     if(keypoints_1[i].pt.x > 0+20 && keypoints_1[i].pt.x < 0+img_1.cols-20
    //        && keypoints_1[i].pt.y > img_1.rows/2+20 && keypoints_1[i].pt.y < img_1.rows/2+img_1.rows/2-20)
    //     {
    //         keypoints_1_temp.push_back(keypoints_1[i]);
    //     }
    // }
    // for(int i = 0; i < keypoints_2.size(); ++i)
    // {
    //     if(keypoints_2[i].pt.x > 0+20 && keypoints_2[i].pt.x < 0+img_2.cols-20
    //        && keypoints_2[i].pt.y > img_2.rows/2+20 && keypoints_2[i].pt.y < img_2.rows/2+img_2.rows/2-20)
    //     {
    //         keypoints_2_temp.push_back(keypoints_2[i]);
    //     }
    // }
    // keypoints_1.clear();
    // keypoints_2.clear();
    // keypoints_1 = keypoints_1_temp;
    // keypoints_2 = keypoints_2_temp;

    //-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
    std::vector<cv::DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- ���Ĳ�:ƥ����ɸѡ
    double min_dist = 10000, max_dist = 0;

    //�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ�������?��ľ���??
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= std::max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

// using namespace std;
// using namespace cv;
int AutoCalib::recoverPose(cv::InputArray &E, cv::InputArray &_points1, cv::InputArray &_points2, cv::OutputArray &_R,
                           cv::OutputArray &_t, double &focal, cv::Point2d &pp, cv::InputOutputArray &_mask)
{
    cv::Mat points1, points2, cameraMatrix;
    cameraMatrix = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
              points1.type() == points2.type());

    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);
    cv::Mat P0 = cv::Mat::eye(3, 4, R1.type());
    cv::Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P1.col(3) = t * 1.0;
    P2(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0;
    P2.col(3) = t * 1.0;
    P3(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P3.col(3) = -t * 1.0;
    P4(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0;
    P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    cv::Mat Q;
    cv::triangulatePoints(P0, P1, points1, points2, Q);
    cv::Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    cv::triangulatePoints(P0, P2, points1, points2, Q);
    cv::Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    cv::triangulatePoints(P0, P3, points1, points2, Q);
    cv::Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    cv::triangulatePoints(P0, P4, points1, points2, Q);
    cv::Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
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
        cv::Mat mask = _mask.getMat();
        CV_Assert(mask.size() == mask1.size());
        cv::bitwise_and(mask, mask1, mask1);
        cv::bitwise_and(mask, mask2, mask2);
        cv::bitwise_and(mask, mask3, mask3);
        cv::bitwise_and(mask, mask4, mask4);
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
        if (_mask.needed())
            mask1.copyTo(_mask);
        return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask2.copyTo(_mask);
        return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask3.copyTo(_mask);
        return good3;
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask4.copyTo(_mask);
        return good4;
    }
}

void AutoCalib::pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                                     std::vector<cv::KeyPoint> keypoints_2,
                                     std::vector<cv::DMatch> matches,
                                     cv::Mat &R, cv::Mat &t)
{
    // ����ڲ�??,TUM Freiburg2
    // Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    // ����ڲ�??
    //float cx = 325.5;
    //float cy = 253.5;
    //float fx = 518.0;
    //float fy = 519.0;
    //float depth_scale = 1000.0;

    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0, 0, 1;

    //-- ��ƥ����?��Ϊvector<Point2f>����ʽ
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- �����������??
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    // cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- ���㱾�ʾ���
    // Point2d principal_point ( 325.1, 249.7 );	//�������??, TUM dataset�궨ֵ
    // double focal_length = 521;			//�������??, TUM dataset�궨ֵ
    cv::Point2d principal_point(325.5, 253.5); //�������??, TUM dataset�궨ֵ
    double focal_length = 519;                 //�������??, TUM dataset�궨ֵ
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    // cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- ���㵥Ӧ����
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    // cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- �ӱ��ʾ����лָ���ת��ƽ����Ϣ.
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "ORB -- R is " << std::endl
              << R << std::endl;
    std::cout << "ORB -- t is " << std::endl
              << t << std::endl;

    Eigen::Matrix3f eigen_rot; // = Eigen::Matrix3f::Identity();
    eigen_rot << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(1, 0),
        R.at<double>(1, 1), R.at<double>(1, 2), R.at<double>(2, 0),
        R.at<double>(2, 1), R.at<double>(2, 2);
    Eigen::Vector3f euler_ang_xyz = eigen_rot.eulerAngles(0, 1, 2); //x y z
    Eigen::Vector3f euler_ang_zyx = eigen_rot.eulerAngles(2, 1, 0); //z y x
    // std::cout << "euler_ang_xyz = " << std::endl << euler_ang_xyz * 180.0 / M_PI << std::endl;
    std::cout << "ORB -- euler_ang_zyx is " << std::endl
              << euler_ang_zyx * 180.0 / M_PI << std::endl;
}

void AutoCalib::send_transform_thread() //û�õ�
{
    tf::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf_trans);
    std::cout << "send transform thread" << std::endl;
}

//�Ե��ƽ��н����������ͼ�����
void AutoCalib::down_sample_pc(pcl::PointCloud<pcl::PointXYZI>::Ptr &in, pcl::PointCloud<pcl::PointXYZI>::Ptr &out, double leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZI> filter;
    filter.setInputCloud(in);
    filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    filter.filter(*out);
}

bool AutoCalib::pointcmp(PointXYZIA a, PointXYZIA b) //�Զ���º���������û�õ�??
{
    return a.cosangle < b.cosangle;
}

//�ӵ�������ȡ�������ĺ������ǳ�����д��һǧ��
void AutoCalib::extract_pc_feature_6(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature)
{
    float factor_t = ((upperBound - lowerBound) / (rings - 1));
    factor = ((rings - 1) / (upperBound - lowerBound));
    YAML::Node local_config = config["extract_pc_edges"];
    float dis_threshold = config["dis_threshold"].as<float>();
    std::vector<std::vector<float>> pc_image; //��ά��Vector
    std::vector<std::vector<float>> pc_image_copy;
    pc_image.resize(1000);
    pc_image_copy.resize(1000); //��һά��1000
    // resize img and set to -1
    for (int i = 0; i < pc_image.size(); i++) //ÿһ���״�����64����
    {
        pc_image[i].resize(rings);
        pc_image_copy[i].resize(rings);
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image[i][j] = -1;
    }
    // convert pointcloud from 3D to 2D img
    for (size_t i = 0; i < pc->size(); i++) //����ֻ�����˵���ת����ͼ��
    {
        float theta = 0;
        if (pc->points[i].y == 0)
            theta = 90.0;
        else if (pc->points[i].y > 0)
        {
            float tan_theta = pc->points[i].x / pc->points[i].y;
            theta = 180 * std::atan(tan_theta) / M_PI;
        }
        else
        {
            float tan_theta = -pc->points[i].y / pc->points[i].x;
            theta = 180 * std::atan(tan_theta) / M_PI;
            theta = 90 + theta;
        }
        int col = cvFloor(theta / 0.18); // theta [0, 180] ==> [0, 1000]
        if (col < 0 || col > 999)
            continue;
        float hypotenuse = std::sqrt(std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].y, 2));
        float angle = std::atan(pc->points[i].z / hypotenuse);
        int ring_id = int(((angle * 180 / M_PI) - lowerBound) * factor + 0.5); //�����еѿ�������ϵ������ϵ�ı任
        if (ring_id < 0 || ring_id > rings - 1)
            continue;
        float dist = std::sqrt(std::pow(pc->points[i].y, 2) + std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].z, 2));
        if (dist < 2)
            continue; //10
        if (pc_image[col][ring_id] == -1)
        {
            pc_image[col][ring_id] = dist; //range
        }
        else if (dist < pc_image[col][ring_id])
        {
            pc_image[col][ring_id] = dist; //set the nearer point
        }
    }

    // // show pc_image by cv::imshow
    // cv::Mat pc_img = cv::Mat::zeros(1000, 64, CV_8UC1);
    // int cnt = 0;
    // float max_range = 0;
    // std::cout<<"0"<<std::endl;
    // for(int i = 0; i < 1000; i++){
    //     for(int j = 0; j < 64; j++){
    //         if((int)pc_image[i][j] > max_range) max_range = (int)pc_image[i][j];
    //         if((int)pc_image[i][j] == -1) pc_image[i][j] = 0;
    //     }
    // }
    // std::cout<<"1"<<std::endl;
    // for(int i = 0; i < 1000; i++){
    //     for(int j = 0; j < 64; j++){
    //         pc_image[i][j] = pc_image[i][j] / max_range * 255;
    //     }
    // }
    // std::cout<<"2"<<std::endl;
    // for(int i = 0; i < 1000; i++){
    //     for(int j = 0; j < 64; j++){
    //         pc_img.at<uchar>(i, j) = (int)pc_image[i][j];//����ĸ�ֵ����??
    //         std::cout << "pc_Img = " << (int)pc_image[i][j] << std::endl;
    //         if((int)pc_image[i][j] != -1) cnt++;
    //     }
    // }
    // std::cout<<"cnt = "<<cnt<<std::endl;
    // cv::namedWindow("pc_img", CV_WINDOW_NORMAL);
    // cv::imshow("pc_img", pc_img);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/pc_img.png", pc_img);
    // cv::waitKey(0);

    // copy
    for (int i = 0; i < pc_image.size(); i++) //��ȡ����������ϸ���룬û��ϸ������700���е�1600����
    {
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image_copy[i][j] = pc_image[i][j];
    }
    for (int i = 1; i < rings - 1; i++)
    {
        for (int j = 1; j < pc_image.size() - 1; j++)
        {
            float sum_dis = 0.0;
            int sum_n = 0;
            float far_sum_dis = 0.0;
            int far_sum_n = 0;
            float near_sum_dis = 0.0;
            int near_sum_n = 0;
            if (pc_image_copy[j - 1][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i - 1] - pc_image[j][i] > dis_threshold)
                    { //������ڵ�ȴ˵�Զ��һ����ֵ
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i - 1] > dis_threshold)
                    { //����˵�����ڵ�Զ��һ����ֵ
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j - 1][i] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i];
                sum_n++;
            }
            if (pc_image_copy[j - 1][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i + 1];
                sum_n++;
            }
            if (pc_image_copy[j][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j][i - 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j][i - 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j][i + 1];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i - 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i - 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i + 1];
                sum_n++;
            }
            if (sum_n >= 5 && pc_image[j][i] == -1)
            {                                     //>=5
                pc_image[j][i] = sum_dis / sum_n; //�����Χ�㶼�в��Ҵ˵��?-1������?ƽ��
                continue;
            }
            if (near_sum_n > sum_n / 2)
            {
                pc_image[j][i] = near_sum_dis / near_sum_n; //�����Χ����ȴ˵��??
            }
            if (far_sum_n > sum_n / 2)
            {
                pc_image[j][i] = far_sum_dis / far_sum_n; //�����Χ����ȴ˵�Զ
            }
        }
    }

    //pc_image data structure
    //  **
    //  **   1000*64
    //  **

    //�����Χ�ĵ㶼�?-1��Ϊ-1
    //   *
    //  *#*
    //   *
    for (int i = 0; i < rings; i++)
    {
        if (i == 0)
        { //������һ��
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                { //����pc_image��һ��
                    if (pc_image[j][i + 1] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1;              //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i]; //����Ϊpc_image
                }
                else if (j == pc_image.size() - 1)
                { //����pc_image���һ��??
                    if (pc_image[j][i + 1] == -1 && pc_image[j - 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i + 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
        else if (i == rings - 1)
        { //�������һ��??
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else if (j == pc_image.size() - 1)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
        else
        {
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j + 1][i] == -1 && pc_image[j][i + 1] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1.��Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else if (j == pc_image.size() - 1)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j][i + 1] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i + 1] == -1 && pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
    }

    for (int i = 0; i < rings; i++)
    {
        for (int j = 1; j < pc_image.size() - 1; j++)
        {
            if (pc_image[j][i] == -1)
            { //����˵��?-1������һ�к���һ�ж���Ϊ-1��ȡ��һ�л���һ�н�С���Ǹ�
                if (pc_image_copy[j - 1][i] != -1 && pc_image_copy[j + 1][i] != -1)
                {
                    pc_image[j][i] = pc_image_copy[j - 1][i] > pc_image_copy[j + 1][i] ? pc_image_copy[j + 1][i] : pc_image_copy[j - 1][i];
                }
            }
        }
    }

    // cv::Mat pc_img = cv::Mat::zeros();

    std::vector<std::vector<float>> mk_rings;
    for (int j = 0; j < pc_image.size(); j++)
    { //1000
        std::vector<float> mk_ring;
        mk_ring.clear();
        for (int i = 0; i < rings; i++)
        {
            // if(pc_image[j][i] != -1)//������в�����??-1�ĵ�
            {
                mk_ring.push_back(pc_image[j][i]); //�洢һ��rings����������
                // mk_ring.push_back(j);
            }
        }
        mk_rings.push_back(mk_ring); //�洢1000����
    }
    // std::cout<<"0"<<std::endl;

    //pc_image data structure  ��ֱ����
    //  **
    //  **   1000*64
    //  **
    for (int i = 0; i < rings; i++) //i<64
    {
        std::vector<float> mk;
        for (int j = 0; j < pc_image.size(); j++)
        { //j<1000
            if (pc_image[j][i] != -1)
            {                                 //������в�����??-1�ĵ�
                mk.push_back(pc_image[j][i]); //�����??(index: 0 2 4 6 ...)
                mk.push_back(j);              //�����?? 0-999(index: 1 3 5 7 ...)
            }
        }
        if (mk.size() < 6)
            continue;

#define DIS 0.1
        for (int j = 1; j < (mk.size() / 2) - 1; j++)
        { //mk��size����һ��(1000)���в�����-1�ĸ�����2
            // if(mk[(j-1)*2]!=-1 && (
            if (
                // mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                // mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())) || // && std::abs(mk[(j-1)*2+1]-mk[(j)*2+1])==1) || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())) || //&& std::abs(mk[(j+1)*2+1]-mk[(j)*2+1])==1) || //ˮƽ�ұߵ����˵����һ�����?(��������)
                // mk[j*2+1] - mk[(j-1)*2+1] > local_config["angle_pixel_dis"].as<int>() || //ˮƽ�ǶȾ������һ�����?
                // mk[(j+1)*2+1] - mk[j*2+1] > local_config["angle_pixel_dis"].as<int>() ||
                local_config["show_all"].as<bool>())
            {

                if (i == 0) // bottom
                {
                    // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // ��
                    //  **
                    // #** ����
                    //  **
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];

                    if ((abs(up - cen) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == 1)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // * **
                    // *#**  ->��
                    // * **
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == rings - 1)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // **
                    // **#  ->��
                    // **
                    if ((abs(cen - dw) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == rings - 2)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // ** *
                    // **#*  ->��
                    // ** *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                // std::cout<<"1"<<std::endl;
                if (i > 1 && i < rings - 2)
                {
                    // std::cout << "mk size = " << mk.size() << std::endl;
                    // std::cout << "mk[(j+1)*2+1]  = " << mk[(j+1)*2+1] << std::endl;
                    // std::cout << "rings size = " << mk_rings.size() << std::endl;
                    // std::cout<<"j i = " << j << " " << i << std::endl;
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // std::cout << "cen = " << cen<<" "<<up << " "<<dw<<" "<<lu<<" "<<ru<<" "<<ld<<" "<<rd<<std::endl;
                    // if(abs(mk_rings[mk[2*j+1]][i-1]-mk_rings[mk[2*j+1]][i])>0.2 || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i])>0.2
                    // ||  abs(mk_rings[mk[2*j+1]][i-2]-mk_rings[mk[2*j+1]][i])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i])>0.24
                    //    || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i-1])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i-2])>0.3
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������ ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // ** **
                    // **#**  ->��
                    // ** **
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS && abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                // recover the point from distance, the layer, and angle of the point
                pcl::PointXYZI p;
                p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180.0) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180.0);
                p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180.0) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180.0);
                p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180.0);
                // i->64  j->1000   mk[dis,j, dis,j, ...]
                //�ж�ˮƽ�����Ƿ����һ����ֵ��intensityȡ��ֵ�����һ���������в�ֵԽ�����Խ������һ����
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
                {
                    // if(mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>()){
                    p.intensity = (mk[(j - 1) * 2] - mk[(j)*2]) > (mk[(j + 1) * 2] - mk[(j)*2]) ? (mk[(j - 1) * 2] - mk[(j)*2]) : (mk[(j + 1) * 2] - mk[(j)*2]); //ȡ��ֵ��ģ������ֵ
                }
                // TODO: ��ֱ����Ĳ�ֵ�浽intensity��
                else if (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1] > local_config["angle_pixel_dis"].as<int>() || mk[(j + 1) * 2 + 1] - mk[j * 2 + 1] > local_config["angle_pixel_dis"].as<int>())
                {
                    p.intensity = (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) > (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]) ? (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) : (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]); //ȡ��ֵ��ģ������ֵ
                }
                pc_feature->push_back(p); //delete horizontal features temply
            }
            else if (local_config["add_edge"].as<bool>())
            {
                if (i != 0 && i != rings - 1)
                {
                    if (pc_image[mk[j * 2 + 1]][i + 1] != -1 && pc_image[mk[j * 2 + 1]][i + 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                    else if (pc_image[mk[j * 2 + 1]][i - 1] != -1 && pc_image[mk[j * 2 + 1]][i - 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                }
            }
        }
    }

    //pc_image data structure  ˮƽ����
    //  **
    //  **   1000*64
    //  **
    for (int i = 0; i < pc_image.size(); i++) //i<1000
    {
        std::vector<float> mk;

        for (int j = 0; j < rings; j++) //j<64
        {
            if (pc_image[i][j] != -1) //������в�����??-1�ĵ�
            {
                mk.push_back(pc_image[i][j]); //�����??(index: 0 2 4 6 ...)
                mk.push_back(j);              //�����?? 0-64(index: 1 3 5 7 ...)
            }                                 //������
        }

        if (mk.size() < 2)
            continue;

#define DIS 0.1
        for (int j = 1; j < (mk.size() / 2) - 1; j++) //mk��size����һ��(64)���в�����-1�ĸ�����2
        {                                             //j=1;j<64-1;j++
            // if(mk[(j-1)*2]!=-1 && (
            if (
                // mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                // mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() * 1.5)) || // && (std::abs(mk[(j-1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() * 1.5)) || //&& (std::abs(mk[(j+1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
                // mk[j*2+1] - mk[(j-1)*2+1] > local_config["angle_pixel_dis"].as<int>() || //ˮƽ�ǶȾ������һ�����?
                // mk[(j+1)*2+1] - mk[j*2+1] > local_config["angle_pixel_dis"].as<int>() ||
                local_config["show_all"].as<bool>())
            {

                // if(j == 0) // bottom
                // {
                //     // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                //     // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                //     //  **
                //     // #**
                //     //  **
                //     float cen = mk_rings[i][mk[2*j+1]];
                //     float up = mk_rings[i+1][mk[2*j+1]];
                //     float lu = mk_rings[i+1][mk[2*j-1]];
                //     float ru = mk_rings[i+1][mk[2*j+3]];
                //     float uu = mk_rings[i+2][mk[2*j+1]];
                //     float luu = mk_rings[i+2][mk[2*j-1]];
                //     float ruu = mk_rings[i+2][mk[2*j+3]];

                //     if( (abs(up-cen)>DIS&&abs(lu-cen)>DIS&&abs(ru-cen)>DIS)
                //         || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS)
                //     )
                //     {
                //         continue;
                //     }
                // }

                //          j=[dis,j,dis,j...]
                //          ***
                // i=1000   ***
                //          ***
                //          ***

                if (j == 1 && i > 0 && i < pc_image.size() - 1)
                {
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float uu = mk_rings[i][mk[2 * j + 5]];
                    float luu = mk_rings[i - 1][mk[2 * j + 5]];
                    float ruu = mk_rings[i + 1][mk[2 * j + 5]];
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    /*// ****
                    // *#**  ->��
                    // *****/

                    // ***
                    // *#*  ->��
                    // ***
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS //��Ҫ���ҵĵ㣬��ΪԶ���򵽵���ĵ�ᱻ���㵽�����ߣ���Ϊ̫Զ�ˣ�����֮�����??����DIS
                         )
                        // || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // if(j == rings-1)
                // {
                //     float cen = mk_rings[i][mk[2*j+1]];
                //     float dw = mk_rings[i-1][mk[2*j+1]];
                //     float ld = mk_rings[i-1][mk[2*j-1]];
                //     float rd = mk_rings[i-1][mk[2*j+3]];
                //     float dd = mk_rings[i-2][mk[2*j+1]];
                //     float ldd = mk_rings[i-2][mk[2*j-1]];
                //     float rdd = mk_rings[i-2][mk[2*j+3]];

                //     // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                //     // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                //     // **
                //     // **#  ->��
                //     // **
                //     if( (abs(cen-dw)>DIS&&abs(ld-cen)>DIS&&abs(rd-cen)>DIS)
                //         || (abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                //     )
                //     {
                //         continue;
                //     }
                // }
                if (j == rings - 2 && i > 0 && i < pc_image.size() - 1)
                {
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float dd = mk_rings[i][mk[2 * j - 3]];
                    float ldd = mk_rings[i - 1][mk[2 * j - 3]];
                    float rdd = mk_rings[i + 1][mk[2 * j - 3]];
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    /* // ** *
                    // **#*  ->��
                    // ** *   */

                    // * *
                    // *#*  ->��
                    // * *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS
                         )
                        // || (abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // std::cout<<"1"<<std::endl;
                if (j > 1 && j < rings - 2 && i > 0 && i < pc_image.size() - 1)
                {
                    // std::cout << "mk size = " << mk.size() << std::endl;
                    // std::cout << "mk[(j+1)*2+1]  = " << mk[(j+1)*2+1] << std::endl;
                    // std::cout << "rings size = " << mk_rings.size() << std::endl;
                    // std::cout<<"j i = " << j << " " << i << std::endl;
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float uu = mk_rings[i][mk[2 * j + 5]];
                    float dd = mk_rings[i][mk[2 * j - 3]];
                    float luu = mk_rings[i - 1][mk[2 * j + 5]];
                    float ruu = mk_rings[i + 1][mk[2 * j + 5]];
                    float ldd = mk_rings[i - 1][mk[2 * j - 3]];
                    float rdd = mk_rings[i + 1][mk[2 * j - 3]];

                    // std::cout << "cen = " << cen<<" "<<up << " "<<dw<<" "<<lu<<" "<<ru<<" "<<ld<<" "<<rd<<std::endl;
                    // if(abs(mk_rings[mk[2*j+1]][i-1]-mk_rings[mk[2*j+1]][i])>0.2 || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i])>0.2
                    // ||  abs(mk_rings[mk[2*j+1]][i-2]-mk_rings[mk[2*j+1]][i])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i])>0.24
                    //    || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i-1])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i-2])>0.3
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������ ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    /*  // ** **
                    // **#**  ->��
                    // ** **  */

                    // * *
                    // *#*  ->��
                    // * *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS
                         )
                        // || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS && abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // i->1000  j->64   mk[dis,j, dis,j, ...]
                // recover the point from distance, the layer, and angle of the point
                pcl::PointXYZI p; //mk[(j)*2]�ǵ�ľ������?
                p.x = mk[(j)*2] * std::cos((j * factor_t + lowerBound) * M_PI / 180) * std::cos((i * 0.18 - 90) * M_PI / 180);
                p.y = mk[(j)*2] * std::cos((j * factor_t + lowerBound) * M_PI / 180) * std::sin(-(i * 0.18 - 90) * M_PI / 180);
                p.z = mk[(j)*2] * std::sin((mk[j * 2 + 1] * factor_t + lowerBound) * M_PI / 180);

                //�ж�ˮƽ�����Ƿ����һ����ֵ��intensityȡ��ֵ�����һ������ֵԽ��Խ������һ�����?
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
                {
                    // if(mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>()){
                    p.intensity = (mk[(j - 1) * 2] - mk[(j)*2]) > (mk[(j + 1) * 2] - mk[(j)*2]) ? (mk[(j - 1) * 2] - mk[(j)*2]) : (mk[(j + 1) * 2] - mk[(j)*2]); //ȡ�󣬴�����ֵ
                }
                //
                else if (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1] > local_config["angle_pixel_dis"].as<int>() || mk[(j + 1) * 2 + 1] - mk[j * 2 + 1] > local_config["angle_pixel_dis"].as<int>())
                {
                    p.intensity = (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) > (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]) ? (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) : (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]); //ȡ�󣬴�����ֵ
                }
                // p.intensity = int(mk[j*2+1] * 30) % 255;
                p.intensity = 0.1;        // TEST: set 0.1 to label horizontal line features
                pc_feature->push_back(p); // not push back ˮƽ����
            }
            else if (local_config["add_edge"].as<bool>())
            {
                if (i != 0 && i != rings - 1)
                {
                    if (pc_image[mk[j * 2 + 1]][i + 1] != -1 && pc_image[mk[j * 2 + 1]][i + 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                    else if (pc_image[mk[j * 2 + 1]][i - 1] != -1 && pc_image[mk[j * 2 + 1]][i - 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                }
            }
        }
    }

    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_cluster(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*pc_feature, *pc_cluster);
    if (pc_cluster->points.size() > 0)
    {
        kdtree->setInputCloud(pc_cluster);
    }
    std::vector<pcl::PointIndices> local_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclid;
    euclid.setInputCloud(pc_cluster);
    euclid.setClusterTolerance(0.3);
    euclid.setMinClusterSize(1);
    euclid.setMaxClusterSize(1000);
    euclid.setSearchMethod(kdtree);
    euclid.extract(local_indices);
    // std::cout<<"local_indices size = " << local_indices.size() << std::endl;

    std::vector<Detected_Obj> obj_list;
    for (size_t i = 0; i < local_indices.size(); i++)
    {
        // the structure to save one detected object
        Detected_Obj obj_info;

        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();

        for (auto pit = local_indices[i].indices.begin(); pit != local_indices[i].indices.end(); ++pit)
        {
            //fill new colored cluster point by point
            pcl::PointXYZ p;
            p.x = pc_feature->points[*pit].x;
            p.y = pc_feature->points[*pit].y;
            p.z = pc_feature->points[*pit].z;

            obj_info.centroid_.x += p.x;
            obj_info.centroid_.y += p.y;
            obj_info.centroid_.z += p.z;

            if (p.x < min_x)
                min_x = p.x;
            if (p.y < min_y)
                min_y = p.y;
            if (p.z < min_z)
                min_z = p.z;
            if (p.x > max_x)
                max_x = p.x;
            if (p.y > max_y)
                max_y = p.y;
            if (p.z > max_z)
                max_z = p.z;
        }

        //min, max points
        obj_info.min_point_.x = min_x;
        obj_info.min_point_.y = min_y;
        obj_info.min_point_.z = min_z;

        obj_info.max_point_.x = max_x;
        obj_info.max_point_.y = max_y;
        obj_info.max_point_.z = max_z;

        //calculate centroid, average
        if (local_indices[i].indices.size() > 0)
        {
            obj_info.centroid_.x /= local_indices[i].indices.size();
            obj_info.centroid_.y /= local_indices[i].indices.size();
            obj_info.centroid_.z /= local_indices[i].indices.size();
        }

        //calculate bounding box
        double length_ = obj_info.max_point_.x - obj_info.min_point_.x;
        double width_ = obj_info.max_point_.y - obj_info.min_point_.y;
        double height_ = obj_info.max_point_.z - obj_info.min_point_.z;

        // obj_info.bounding_box_.header = "object";

        obj_info.bounding_box_.pose.position.x = obj_info.min_point_.x + length_ / 2;
        obj_info.bounding_box_.pose.position.y = obj_info.min_point_.y + width_ / 2;
        obj_info.bounding_box_.pose.position.z = obj_info.min_point_.z + height_ / 2;

        obj_info.bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
        obj_info.bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
        obj_info.bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

        if (obj_info.bounding_box_.dimensions.x > 0.5 || obj_info.bounding_box_.dimensions.y > 0.5 || obj_info.bounding_box_.dimensions.z > 0.5)
        {
            obj_list.push_back(obj_info);
        }
    }
    // std::cout<<"obj list size = " << obj_list.size() << std::endl;
    // for(size_t i = 0; i < obj_list.size(); ++i)
    // {
    //     std::cout<<"obj list xyz = " <<obj_list[i].bounding_box_.pose.position.x<<" " <<obj_list[i].bounding_box_.pose.position.y<<
    //     " "<< obj_list[i].bounding_box_.pose.position.z << " " <<obj_list[i].bounding_box_.dimensions.x << " " << obj_list[i].bounding_box_.dimensions.y << " "
    //      << obj_list[i].bounding_box_.dimensions.z << std::endl;
    // }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_cluster(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t j = 0; j < obj_list.size(); ++j)
    {
        float x_min = obj_list[j].bounding_box_.pose.position.x - obj_list[j].bounding_box_.dimensions.x / 2;
        float x_max = obj_list[j].bounding_box_.pose.position.x + obj_list[j].bounding_box_.dimensions.x / 2;
        float y_min = obj_list[j].bounding_box_.pose.position.y - obj_list[j].bounding_box_.dimensions.y / 2;
        float y_max = obj_list[j].bounding_box_.pose.position.y + obj_list[j].bounding_box_.dimensions.y / 2;
        float z_min = obj_list[j].bounding_box_.pose.position.z - obj_list[j].bounding_box_.dimensions.z / 2;
        float z_max = obj_list[j].bounding_box_.pose.position.z + obj_list[j].bounding_box_.dimensions.z / 2;
        for (size_t i = 0; i < pc_feature->points.size(); ++i)
        {
            float pc_x = pc_feature->points[i].x;
            float pc_y = pc_feature->points[i].y;
            float pc_z = pc_feature->points[i].z;
            if (pc_x > x_min - 0.1 && pc_x < x_max + 0.1 && pc_y > y_min - 0.1 && pc_y < y_max + 0.1 && pc_z > z_min - 0.1 && pc_z < z_max + 0.1)
            {
                pcl::PointXYZI p;
                p.x = pc_x;
                p.y = pc_y;
                p.z = pc_z;
                p.intensity = pc_feature->points[i].intensity;
                // std::cout<<"xyz = "<<p.x<< " " <<  p.y << " " << p.z<<std::endl;
                pc_feature_cluster->push_back(p);
                // std::cout<<"after"<<std::endl;
            }
        }
    }
    if (config["cluster_pointcloud"].as<bool>())
    {
        pcl::copyPointCloud(*pc_feature_cluster, *pc_feature); // whether use the cluster method
    }

    // ros::Publisher pub_;
    // pub_ = nh.advertise<sensor_msgs::PointCloud2>("pc_feature_cluster", 50);

    // sensor_msgs::PointCloud2 cloud_msg;
    // pcl::toROSMsg(*pc_feature_cluster, cloud_msg);
    // cloud_msg.header.stamp = ros::Time::now();
    // cloud_msg.header.frame_id = "velodyne";

    // ros::Rate loop_rate(50);
    // while(ros::ok())
    // {
    //     ros::spinOnce();
    //     pub_.publish(cloud_msg);
    //     std::cout << "publish" << std::endl;
    //     loop_rate.sleep();
    // }

    //  //��ʾ��ȡЧ��
    // pcl::visualization::PCLVisualizer viewer("pc Viewer");
    // pcl::visualization::PCLVisualizer viewer_feature("pc_feature Viewer");
    // pcl::visualization::PCLVisualizer viewer_feature_cluster("pc_feature Viewer Cluster");
    // //���ô��ڱ�����ɫ����ΧΪ0-1
    // viewer.setBackgroundColor(0, 0, 0);
    // viewer_feature.setBackgroundColor(0, 0, 0);
    // viewer_feature_cluster.setBackgroundColor(0, 0, 0);
    // //����������
    // viewer.addCoordinateSystem(1);
    // viewer_feature.addCoordinateSystem(1);
    // viewer_feature_cluster.addCoordinateSystem(1);
    // //���ݵ�����ĳ���ֶδ�С������ɫ
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(pc, "z");
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor_feature(pc_feature, "z");
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor_feature_cluster(pc_feature_cluster, "z");
    // //���������ӵ��Ʋ�������ɫ
    // viewer.addPointCloud(pc, fildColor, "cloud");
    // viewer_feature.addPointCloud(pc_feature, fildColor_feature, "cloud_feature");
    // viewer_feature_cluster.addPointCloud(pc_feature_cluster, fildColor_feature, "cloud_feature_cluster");
    // //���ӵ��ƺ�ͨ������ID��������ʾ��С
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
    // viewer_feature.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_feature");
    // viewer_feature_cluster.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_feature_cluster");
    // //�����������������ʾ������??
    // viewer.resetCamera();
    // viewer_feature.resetCamera();
    // viewer_feature_cluster.resetCamera();
    // while (!viewer.wasStopped())
    // {
    //   viewer.spinOnce();
    //   viewer_feature.spinOnce();
    //   viewer_feature_cluster.spinOnce();
    // }
}

float AutoCalib::dist2Point(int x1, int y1, int x2, int y2) //������֮��ľ���??
{
    return std::sqrt(double(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

//��ȡͼ��������������õ�opencv���lsd����
void AutoCalib::extract_image_feature(cv::Mat &img, cv::Mat &image2, std::vector<cv::line_descriptor::KeyLine> &keylines, std::vector<cv::line_descriptor::KeyLine> &keylines2,
                                      cv::Mat &outimg, cv::Mat &outimg_thin)
{
    cv::Mat mLdesc, mLdesc2;
    std::vector<std::vector<cv::DMatch>> lmatches;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

    // std::cout<<"img channels = " << img.channels() << std::endl;
    // std::cout << (img.type() == CV_8UC1) << std::endl;
    // if(img.channels()==1)
    // {
    //     // cv::cvtColor(raw_image, output_image, cv::COLOR_GRAY2BGR);
    //     cv::Mat img_temp(img.rows, img.cols, CV_8UC3, cv::Scalar::all(0));
    //     for (int i = 0; i < img.cols; i++)
    //     {
    //         for (int j = 0; j < img.rows; j++)
    //         {
    //             img_temp.at<cv::Vec3b>(j,i)[2] = img_temp.at<cv::Vec3b>(j,i)[1] =
    //             img_temp.at<cv::Vec3b>(j,i)[0] = (int) img.at<uchar>(j, i);
    //         }
    //     }
    //     img_temp.copyTo(img);
    //     // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
    //     // cv::imshow("cvt", output_image);
    //     // cv::waitKey(0);
    //     // std::cout << "after cvt" << std::endl;
    // }
    // if(image2.channels()==1)
    // {
    //     // cv::cvtColor(raw_image, output_image, cv::COLOR_GRAY2BGR);
    //     cv::Mat img_temp(image2.rows, image2.cols, CV_8UC3, cv::Scalar::all(0));
    //     for (int i = 0; i < image2.cols; i++)
    //     {
    //         for (int j = 0; j < image2.rows; j++)
    //         {
    //             img_temp.at<cv::Vec3b>(j,i)[2] = img_temp.at<cv::Vec3b>(j,i)[1] =
    //             img_temp.at<cv::Vec3b>(j,i)[0] = (int) image2.at<uchar>(j, i);
    //         }
    //     }
    //     img_temp.copyTo(image2);
    //     // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
    //     // cv::imshow("cvt", output_image);
    //     // cv::waitKey(0);
    //     // std::cout << "after cvt" << std::endl;
    // }

    // std::cout << (img.type() == CV_8UC1) << std::endl;

    lsd->detect(img, keylines, 1.2, 1);
    lsd->detect(image2, keylines2, 1.2, 1);
    int lsdNFeatures = 50;
    if (keylines.size() > lsdNFeatures)
    {
        std::sort(keylines.begin(), keylines.end(), [](const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b)
                  { return a.response > b.response; });
        keylines.resize(lsdNFeatures);
        for (int i = 0; i < lsdNFeatures; i++)
            keylines[i].class_id = i;
    }
    if (keylines2.size() > lsdNFeatures)
    {
        std::sort(keylines2.begin(), keylines2.end(), [](const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b)
                  { return a.response > b.response; });
        keylines2.resize(lsdNFeatures);
        for (int i = 0; i < lsdNFeatures; i++)
            keylines2[i].class_id = i;
    }
    // cv::Mat drawLines(img);
    // lsd->drawSegments(drawLines, keylines);
    // cv::imshow("lsd", drawLines);
    // cv::waitKey(0);

    // Create and LSD detector with standard or no refinement.
#if 0
    cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);//��������LSD�㷨������õ���standard��
#else
    cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
#endif
    std::vector<cv::Vec4f> lines_std;
    // Detect the lines
    cv::Mat img_gray;
    if (img.channels() == 1)
    {
        img.copyTo(img_gray);
    }
    else
    {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    }

    ls->detect(img_gray, lines_std); //����Ѽ�⵽��ֱ���߶ζ�������lines_std�У�4��float��ֵ���ֱ�Ϊ��ֹ�������??
    // Show found lines
    // cv::Mat drawnLines(img);
    // ls->drawSegments(drawnLines, lines_std);

    // cv::Mat img_draw = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    cv::Mat img_draw = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar::all(0));
    // cv::Mat::zeros(1000, 64, CV_8UC1);
    // for(int i = 0; i < img.cols; ++i)
    // {
    //     for(int j =0; j < img.rows; ++j)
    //     {
    //             if (img.at<uchar>(j, i) < 0 || img.at<uchar>(j, i) > 255) {
    //                 std::cout << "error" << std::endl;
    //                 exit(0);
    //             }
    //             img.at<uchar>(j, i) = 0;//255 - (int) edge_distance_image2.at<uchar>(j, i);
    //     }
    // }
    //draw
    for (int i = 0; i < lines_std.size(); ++i)
    {
        if (dist2Point(lines_std[i][0], lines_std[i][1], lines_std[i][2], lines_std[i][3]) > 7)
            cv::line(img_draw, cv::Point(lines_std[i][0], lines_std[i][1]), cv::Point(lines_std[i][2], lines_std[i][3]), cv::Scalar(255, 255, 255), 1, CV_AA);
    }
    // cv::imshow("Standard refinement", drawnLines);
    // cv::imshow("Standard refinement", img_draw);
    // std::cout<<"img draw " << img_draw.channels() << " " << img_draw.type() << std::endl;
    cv::threshold(img_draw, img_draw, 100, 255, cv::THRESH_BINARY);
    // cv::waitKey(0);

    static int m = 0;
    std::string m_string = std::to_string(m);

    std::string line_feature_string = "../data" + std::to_string(frame_cnt) + "/result/line_feature_" + m_string + ".png";
    cv::imwrite(line_feature_string, img_draw);
    m++;

    cv::Mat img_draw_copy(img_draw.rows, img_draw.cols, CV_8UC1, cv::Scalar::all(0));
    for (int i = 0; i < img_draw.cols; i++)
    {
        for (int j = 0; j < img_draw.rows; j++)
        {
            if (i >= 5 && j >= 5 && i <= (img_draw.cols - 5) && j <= (img_draw.rows - 5))
            {
                float cen = (int)img_draw.at<uchar>(j, i);
                float up = (int)img_draw.at<uchar>(j - 1, i);
                float dw = (int)img_draw.at<uchar>(j + 1, i);
                float lt = (int)img_draw.at<uchar>(j, i - 1);
                float rt = (int)img_draw.at<uchar>(j, i + 1);
                float lu = (int)img_draw.at<uchar>(j - 1, i - 1);
                float ru = (int)img_draw.at<uchar>(j - 1, i + 1);
                float ld = (int)img_draw.at<uchar>(j + 1, i - 1);
                float rd = (int)img_draw.at<uchar>(j + 1, i + 1);
                float uu = (int)img_draw.at<uchar>(j - 2, i);
                float dd = (int)img_draw.at<uchar>(j + 2, i);
                float luu = (int)img_draw.at<uchar>(j - 2, i - 1);
                float ruu = (int)img_draw.at<uchar>(j - 2, i + 1);
                float ldd = (int)img_draw.at<uchar>(j + 2, i - 1);
                float rdd = (int)img_draw.at<uchar>(j + 2, i + 1);
                float lluu = (int)img_draw.at<uchar>(j - 2, i - 2);
                float rruu = (int)img_draw.at<uchar>(j - 2, i + 2);
                float llu = (int)img_draw.at<uchar>(j - 1, i - 2);
                float rru = (int)img_draw.at<uchar>(j - 1, i + 2);
                float ll = (int)img_draw.at<uchar>(j, i - 2);
                float rr = (int)img_draw.at<uchar>(j, i + 2);
                float lld = (int)img_draw.at<uchar>(j + 1, i - 2);
                float rrd = (int)img_draw.at<uchar>(j + 1, i + 2);
                float lldd = (int)img_draw.at<uchar>(j + 2, i - 2);
                float rrdd = (int)img_draw.at<uchar>(j + 2, i + 2);

                // ��ֱ����
                if (cen == 255 && (lu == 255 || up == 255 || ru == 255 || ld == 255 || dw == 255 || rd == 255) && lluu == 0 && llu == 0 && ll == 0 && lld == 0 && lldd == 0 && rruu == 0 && rru == 0 && rr == 0 && rrd == 0 && rrdd == 0)
                {
                    img_draw_copy.at<uchar>(j, i) = (int)img_draw.at<uchar>(j, i);
                }

                // // ˮƽ����
                if (cen == 255 && (lu == 255 || lt == 255 || ld == 255 || rt == 255 || ru == 255 || rd == 255 || rru == 255 || rr == 255 || rrd == 255) && lluu == 0 && luu == 0 && uu == 0 && ruu == 0 && rruu == 0 && lldd == 0 && ldd == 0 && dd == 0 && rdd == 0 && rrdd == 0)
                {
                    img_draw_copy.at<uchar>(j, i) = (int)img_draw.at<uchar>(j, i);
                }
            }
        }
    }
    // cv::imshow("canny_filter", image_edge_copy);
    // cv::waitKey(0);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter.png", image_edge_copy);

    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/img_draw_copy.png", img_draw_copy);

    cv::Mat img_draw_copy_copy(img_draw.rows, img_draw.cols, CV_8UC1, cv::Scalar::all(0));
    for (int i = 0; i < img_draw.cols; i++)
    {
        for (int j = 0; j < img_draw.rows; j++)
        {
            if (i >= 5 && j >= 5 && i <= (img_draw.cols - 5) && j <= (img_draw.rows - 5))
            {
                float cen = (int)img_draw_copy.at<uchar>(j, i);
                float up = (int)img_draw_copy.at<uchar>(j - 1, i);
                float dw = (int)img_draw_copy.at<uchar>(j + 1, i);
                float lt = (int)img_draw_copy.at<uchar>(j, i - 1);
                float rt = (int)img_draw_copy.at<uchar>(j, i + 1);
                float lu = (int)img_draw_copy.at<uchar>(j - 1, i - 1);
                float ru = (int)img_draw_copy.at<uchar>(j - 1, i + 1);
                float ld = (int)img_draw_copy.at<uchar>(j + 1, i - 1);
                float rd = (int)img_draw_copy.at<uchar>(j + 1, i + 1);

                if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lu == 0 && ru == 0 && ld == 0 && rd == 0)
                {
                    img_draw_copy_copy.at<uchar>(j, i) = 0;
                }
                else
                {
                    img_draw_copy_copy.at<uchar>(j, i) = (int)img_draw_copy.at<uchar>(j, i);
                }
            }
        }
    }
    // cv::imshow("canny_filter_filter", image_edge_copy_copy);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter_filter.png", image_edge_copy_copy);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/img_draw_copy_copy.png", img_draw_copy_copy);

    cv::Mat img_draw_copy_copy_copy(img_draw.rows, img_draw.cols, CV_8UC1, cv::Scalar::all(0));
    for (int i = 0; i < img_draw_copy_copy.cols; i++)
    {
        for (int j = 0; j < img_draw_copy_copy.rows; j++)
        {
            if (i >= 5 && j >= 5 && i <= (img_draw_copy_copy.cols - 5) && j <= (img_draw_copy_copy.rows - 5))
            {
                float cen = (int)img_draw_copy_copy.at<uchar>(j, i);
                float up = (int)img_draw_copy_copy.at<uchar>(j - 1, i);
                float dw = (int)img_draw_copy_copy.at<uchar>(j + 1, i);
                float lt = (int)img_draw_copy_copy.at<uchar>(j, i - 1);
                float rt = (int)img_draw_copy_copy.at<uchar>(j, i + 1);
                float lu = (int)img_draw_copy_copy.at<uchar>(j - 1, i - 1);
                float ru = (int)img_draw_copy_copy.at<uchar>(j - 1, i + 1);
                float ld = (int)img_draw_copy_copy.at<uchar>(j + 1, i - 1);
                float rd = (int)img_draw_copy_copy.at<uchar>(j + 1, i + 1);

                if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lu == 0 && ru == 0 && ld == 0 && rd == 0)
                {
                    img_draw_copy_copy_copy.at<uchar>(j, i) = 0;
                }
                else
                {
                    img_draw_copy_copy_copy.at<uchar>(j, i) = (int)img_draw_copy_copy.at<uchar>(j, i);
                }
            }
        }
    }
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/img_draw_copy_copy_copy.png", img_draw_copy_copy_copy);

    cv::Mat line_temp = cv::Mat(cv::Size(img.rows, img.cols), CV_8UC1, cv::Scalar::all(0));
    // std::cout<<"linetemp0 type channel " << line_temp.type() << "  " << line_temp.channels() << std::endl;
    // for(int i = 0; i < img_draw.cols; i++)
    // {
    //     for(int j =0; j < img_draw.rows; j++)
    //     {
    //         line_temp.at<uchar>(j, i) = (int)img_draw.at<uchar>(j,i);//255 - (int) edge_distance_image2.at<uchar>(j, i);
    //     }
    // }
    if (config["filter_line_features"].as<bool>())
    {
        line_temp = img_draw_copy_copy_copy.clone();
    }
    else
    {
        line_temp = img_draw.clone();
    }

    // cv::imshow("img_draw_copy_copy_copy", img_draw_copy_copy_copy);
    // cv::imshow("img_draw", img_draw);
    // cv::imshow("line_temp", line_temp);
    // cv::waitKey(0);
    // cv::cvtColor(img_draw, line_temp, cv::COLOR_BGR2GRAY);

    // cv::Mat image_edge_copy_copy_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));

    cv::Mat line_img(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));          // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img2(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));         // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img2_bitwise(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0)); // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img3(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));         // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img_thin(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));     // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);

    //ԭ���ǣ���lsd���ٻҶȻ��������仯��
    cv::bitwise_not(line_temp, line_temp);                            //bitwise_not�ǶԶ��������ݽ��С��ǡ�����������ͼ�񣨻Ҷ�ͼ����ɫͼ����ɣ�ÿ������ֵ���ж����ơ��ǡ�������~1=0��~0=1 �� ��ͼƬ������ֵ��λ����
    cv::threshold(line_temp, line_temp, 100, 255, cv::THRESH_BINARY); //��ֵ������ https://blog.csdn.net/u012566751/article/details/77046445
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_temp.png", line_temp);
    // cv::imshow("line_temp", line_temp);
    // cv::imshow("line_img", line_img);
    // std::cout<<"line_temp type " << line_temp.type() << " " << line_temp.channels() << std::endl;
    cv::distanceTransform(line_temp, line_img2, CV_DIST_C, 3); //Method1  ���ǲ��������仯����
    // line_img2.convertTo(line_img2, CV_8UC1, 255);
    // cv::imshow("line_img2", line_img2);
    // cv::distanceTransform(img, line_img, CV_DIST_L2, 5);// Method2
    // cv::imshow("img", img);
    // cv::imshow("line_img", line_img);
    // cv::imshow("line_img2 before", line_img2);
    // cv::bitwise_not(line_img2, line_img2);
    for (int i = 0; i < line_img2.cols; i++)
    {
        for (int j = 0; j < line_img2.rows; j++)
        {
            //         newBImgData[i*step+j] = 255- line_img2[i*step+j];
            line_img2_bitwise.at<uchar>(j, i) = 255 - (int)line_img2.at<uchar>(j, i);
            //         // if(((int)line_img2.at<uchar>(j, i)) > 200) line_img2.at<uchar>(j, i) = 0;
            //         // if(((int)line_img2.at<uchar>(j, i)) < 50) line_img2.at<uchar>(j, i) = 255;
        }
    }
    // line_img2_bitwise = 255 - line_img2;
    // cv::cvtColor(line_img2, line_img2, cv::COLOR_GRAY2BGR);//commont
    // cv::threshold(line_img2, line_img2, 200,255,CV_THRESH_BINARY);
    // cv::bitwise_not(line_img2, line_img2_bitwise);
    cv::normalize(line_img2, line_img3, normalize_config, 0, cv::NORM_INF, 1); // Method1
    // cv::imshow("line_img3", line_img3);
    cv::normalize(line_img2, line_img_thin, normalize_config_thin, 0, cv::NORM_INF, 1); // Method1
    // cv::imshow("line_img_thin", line_img_thin);
    // cv::waitKey(0);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img3before.png", line_img3);
    cv::Mat line_img3_after(line_img3.rows, line_img3.cols, CV_8UC1, cv::Scalar::all(0));     // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img_thin_after(line_img3.rows, line_img3.cols, CV_8UC1, cv::Scalar::all(0)); // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    for (int i = 0; i < line_img3.cols; i++)
    {
        for (int j = 0; j < line_img3.rows; j++)
        {
            if (line_img3.at<uchar>(j, i) < 0 || line_img3.at<uchar>(j, i) > 255)
            {
                std::cout << "error" << std::endl;
                exit(0);
            }
            line_img3_after.at<uchar>(j, i) = 255 - (int)line_img3.at<uchar>(j, i);
            line_img_thin_after.at<uchar>(j, i) = 255 - (int)line_img_thin.at<uchar>(j, i);
            // std::cout << (int)gray_image.at<uchar>(j, i) << std::endl;
        }
    }
    // cv::GaussianBlur(line_img, line_img2, cv::Size(3, 3), 1, 1);// 2, 2
    // cv::normalize(line_img, line_img2, 0, 5., NORM_MINMAX); // Method2
    // cv::imshow("line_img2_bitwise", line_img2_bitwise);
    // cv::imshow("line_img3 after", line_img3);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img2.png", line_img2);

    //cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img.png", line_img3_after);
    //cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img_thin.png", line_img_thin_after);
    outimg = line_img3_after.clone();          //�Ƚϴֵ�������ͼ
    outimg_thin = line_img_thin_after.clone(); //�Ƚ�ϸ��������ͼ

    //cv::imshow("outimg", outimg);
    //cv::imshow("outimg_thin", outimg_thin);
    //cv::Mat outimg_1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));// = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    //cv::Mat outimg_thin_1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    //cv::Mat structureElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7), cv::Point(-1, -1));
    //cv::erode(outimg, outimg_1, structureElement);               //���ø�ʴAPI
    //cv::imshow("��ʴ������", outimg_1);
    //cv::erode(outimg_thin, outimg_thin_1, structureElement);               //���ø�ʴAPI
    //cv::imshow("��ʴ������1��", outimg_thin_1);
    //cv::dilate(outimg_1, outimg, structureElement, cv::Point(-1, -1), 1);               //��������API
    //cv::imshow("���Ͳ�����", outimg);
    //cv::dilate(outimg_thin_1, outimg_thin, structureElement, cv::Point(-1, -1), 1);               //��������API
    //cv::imshow("���Ͳ�����", outimg_thin_1);

    // outimg = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC3);
    // cv::cvtColor(line_img3, outimg, cv::COLOR_GRAY2BGR);
    // cv::imshow("outimg", outimg);
    // cv::waitKey(0);

    //#pragma omp parallel for
    // for (int i = 0; i < edge_distance_image2.cols; i++) {
    //     for (int j = 0; j < edge_distance_image2.rows; j++) {
    //         if (edge_distance_image2.at<uchar>(j, i) < 0 || edge_distance_image2.at<uchar>(j, i) > 255) {
    //             std::cout << "error" << std::endl;
    //             exit(0);
    //         }
    //         gray_image.at<uchar>(j, i) = 255 - (int) edge_distance_image2.at<uchar>(j, i);
    //         // std::cout << (int)gray_image.at<uchar>(j, i) << std::endl;
    //     }
    // }

    // lbd->compute(img, keylines, mLdesc);
    // lbd->compute(image2,keylines2,mLdesc2);
    // cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
    // bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
    // std::vector<cv::DMatch> matches;
    // for(size_t i=0;i<lmatches.size();i++)
    // {
    //     const cv::DMatch& bestMatch = lmatches[i][0];
    //     const cv::DMatch& betterMatch = lmatches[i][1];
    //     float  distanceRatio = bestMatch.distance / betterMatch.distance;
    //     if (distanceRatio < 0.7)
    //         matches.push_back(bestMatch);
    // }

    // cv::Mat outImg;
    // std::vector<char> mask( lmatches.size(), 1 );
    // drawLineMatches( img, keylines, image2, keylines2, matches, outImg, cv::Scalar::all( -1 ), cv::Scalar::all( -1 ), mask, cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT );
    // cv::imshow( "Matches", outImg );
    // cv::waitKey(0);
}
//�ѵ���ͶӰ��ͼ����
void AutoCalib::project2image(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, cv::Mat raw_image, cv::Mat &output_image, Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{

    Eigen::Matrix<float, 3, 4> T_lidar2cam_top3_local, T_lidar2image_local; //lida2image=T_lidar2cam*(T_cam02cam2)*T_cam2image
    T_lidar2cam_top3_local = RT.topRows(3);                                 //R T��ǰ����
    T_lidar2image_local = camera_param * T_lidar2cam_top3_local;
    if (raw_image.channels() < 3 && raw_image.channels() >= 1)
    {
        // std::cout << "before cvt" << std::endl;
        // cv::cvtColor(raw_image, output_image, cv::COLOR_GRAY2BGR);
        cv::Mat output_image_3channels(raw_image.rows, raw_image.cols, CV_8UC3, cv::Scalar::all(0));
        for (int i = 0; i < raw_image.cols; i++)
        {
            for (int j = 0; j < raw_image.rows; j++)
            {
                output_image_3channels.at<cv::Vec3b>(j, i)[2] = output_image_3channels.at<cv::Vec3b>(j, i)[1] =
                    output_image_3channels.at<cv::Vec3b>(j, i)[0] = (int)raw_image.at<uchar>(j, i);
                //  (int) raw_image.at<uchar>(j, i);
                // output_image_3channels.at<cv::Vec3b>(j,i)[0] = (int) raw_image.at<uchar>(j, i);
            }
        }
        output_image_3channels.copyTo(output_image);
        // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
        // cv::imshow("cvt", output_image);
        // cv::waitKey(0);
        // std::cout << "after cvt" << std::endl;
    }
    else
    {
        raw_image.copyTo(output_image);
    }
    pcl::PointXYZI r;
    Eigen::Vector4f raw_point;
    Eigen::Vector3f trans_point;
    double deep, deep_config; //deep_config: normalize, max deep
    int point_r;
    deep_config = 80;
    point_r = 2;
    //std::cout << "image size; " << raw_image.cols << " * " << raw_image.rows << std::endl;
    for (int i = 0; i < pc->size(); i++)
    {
        r = pc->points[i];
        raw_point(0, 0) = r.x; //����˹�һ��ƽ���ϵĵ�??
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point = T_lidar2image_local * raw_point;
        int x = (int)(trans_point(0, 0) / trans_point(2, 0));
        int y = (int)(trans_point(1, 0) / trans_point(2, 0));

        //cout<<"!!!@@@####"<<x<<" "<<y<<" ";

        if (x < 0 || x > (raw_image.cols - 1) || y < 0 || y > (raw_image.rows - 1))
            continue;
        deep = trans_point(2, 0) / deep_config;
        //deep = r.intensity / deep_config;
        int blue, red, green;
        if (deep <= 0.5)
        {
            green = (int)((0.5 - deep) / 0.5 * 255);
            red = (int)(deep / 0.5 * 255);
            blue = 0;
        }
        else if (deep <= 1)
        {
            green = 0;
            red = (int)((1 - deep) / 0.5 * 255);
            blue = (int)((deep - 0.5) / 0.5 * 255);
        }
        else
        {
            blue = 0;
            green = 0;
            red = 255;
        };
        //��ͼ���ϻ�С԰��������ɫ
        //cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(255, 255, 0), -1);
        cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(0, 255, 0), -1);
        // cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(blue,green,red), -1);
    }
}

//��ȡ�����ĺ���������ȥ���ƺ�ͼ�񣬵õ��˵��Ƶ�������
void AutoCalib::extractFeature(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs, std::vector<cv::Mat> images, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pc_feature,
                               std::vector<cv::Mat> &distance_image, std::vector<cv::Mat> &distance_image_thin)
{
    std::cout << "Start extract feature" << std::endl;
    int data_num = pcs.size();
    if (data_num != images.size())
    {
        std::cout << "\033[31mExtractFeatur Error: pointcloud num unequal to image num!\033[0m" << std::endl;
        std::exit(0);
    }

    float search_r, search_r2, search_r3;
    float x_max, x_min, y_min, y_max, z_min, z_max;
    int search_num, search_num2, search_num3;
    x_max = config["x_max"].as<float>();
    x_min = config["x_min"].as<float>();
    y_min = config["y_min"].as<float>();
    y_max = config["y_max"].as<float>();
    z_min = config["z_min"].as<float>();
    z_max = config["z_max"].as<float>();
    search_r = config["search_r"].as<float>();
    search_num = config["search_num"].as<int>();
    search_r2 = config["search_r2"].as<float>();
    search_num2 = config["search_num2"].as<int>();
    search_r3 = config["search_r3"].as<float>();
    search_num3 = config["search_num3"].as<int>();
    dis_threshold = config["dis_threshold"].as<float>();
    angle_threshold = config["angle_threshold"].as<float>();
    canny_threshold_mini = config["canny_threshold_mini"].as<int>();
    canny_threshold_max = config["canny_threshold_max"].as<int>();
    normalize_config = config["normalize_config"].as<int>();
    normalize_config_thin = config["normalize_config_thin"].as<int>();
    factor = ((rings - 1) / (upperBound - lowerBound));

    for (int i = 0; i < data_num; i++)
    {

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw = pcs[i];
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr edges(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*raw, *pc);
        pcl::PassThrough<pcl::PointXYZI> pass_filter;
        pass_filter.setInputCloud(pc);
        pass_filter.setFilterFieldName("y");
        pass_filter.setFilterLimitsNegative(false);
        pass_filter.setFilterLimits(y_min, y_max);
        pass_filter.filter(*filtered_y);

        pass_filter.setInputCloud(filtered_y);
        pass_filter.setFilterFieldName("x");
        pass_filter.setFilterLimitsNegative(false);
        pass_filter.setFilterLimits(x_min, x_max); //x����Сֵ ���ֵ����config�ļ���ֻ����3(min)-max�׷�Χ�ڵĵ��Ƶ�
        pass_filter.filter(*filtered_x);

        pass_filter.setInputCloud(filtered_x);
        pass_filter.setFilterFieldName("z");
        pass_filter.setFilterLimitsNegative(false);
        pass_filter.setFilterLimits(z_min, z_max);
        pass_filter.filter(*filtered);

        filtered_pc.push_back(filtered);

        cv::Mat image1 = images[i];
        cv::Mat image = images[i];
        images_withouthist.push_back(image1);

        if (image1.channels() > 1)
        {
            cv::Mat imageRGB1[3];
            split(image1, imageRGB1);

            // cv::imshow("image_clone0", image_clone1);
            // cv::waitKey(0);

            for (int i = 0; i < 3; i++)
            {
                cv::equalizeHist(imageRGB1[i], imageRGB1[i]);
            }
            cv::merge(imageRGB1, 3, image1);
            //��һ����ǿ
            cv::Mat imageRGB[3];
            split(image1, imageRGB);
            for (int i = 0; i < 3; i++)
            {
                cv::equalizeHist(imageRGB[i], imageRGB[i]);
            }
            cv::merge(imageRGB, 3, image);
        }

        // imshow("ֱ��ͼ���⻯ͼ����ǿЧ��", image);
        // cv::waitKey(0);

        // cv::Mat imageEnhance;
        // cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
        // cv::filter2D(image_clone, imageEnhance, CV_8UC3, kernel);
        // cv::imshow("������˹����ͼ����ǿЧ��", imageEnhance);
        // cv::imshow("image_clone1", image_clone);
        // cv::waitKey(0);

        // cv::Mat imageLog(image_clone.size(), CV_32FC3);
        // for (int i = 0; i < image.rows; i++)
        // {
        //     for (int j = 0; j < image.cols; j++)
        //     {
        //         imageLog.at<cv::Vec3f>(i, j)[0] = log(1 + image_clone.at<cv::Vec3b>(i, j)[0]);
        //         imageLog.at<cv::Vec3f>(i, j)[1] = log(1 + image_clone.at<cv::Vec3b>(i, j)[1]);
        //         imageLog.at<cv::Vec3f>(i, j)[2] = log(1 + image_clone.at<cv::Vec3b>(i, j)[2]);
        //     }
        // }
        // //��һ����0~255
        // cv::normalize(imageLog, imageLog, 0, 255, CV_MINMAX);
        // //ת����8bitͼ����ʾ
        // cv::convertScaleAbs(imageLog, imageLog);
        // cv::imshow("LOGͼ����ǿЧ��", imageLog);
        // cv::imshow("image_clone2", image_clone);
        // cv::waitKey(0);

        // cv::Mat imageGamma(image_clone.size(), CV_32FC3);
        // for (int i = 0; i < image.rows; i++)
        // {
        //     for (int j = 0; j < image.cols; j++)
        //     {
        //         imageGamma.at<cv::Vec3f>(i, j)[0] = (image_clone.at<cv::Vec3b>(i, j)[0])*(image_clone.at<cv::Vec3b>(i, j)[0])*(image_clone.at<cv::Vec3b>(i, j)[0]);
        //         imageGamma.at<cv::Vec3f>(i, j)[1] = (image_clone.at<cv::Vec3b>(i, j)[1])*(image_clone.at<cv::Vec3b>(i, j)[1])*(image_clone.at<cv::Vec3b>(i, j)[1]);
        //         imageGamma.at<cv::Vec3f>(i, j)[2] = (image_clone.at<cv::Vec3b>(i, j)[2])*(image_clone.at<cv::Vec3b>(i, j)[2])*(image_clone.at<cv::Vec3b>(i, j)[2]);
        //     }
        // }
        // //��һ����0~255
        // cv::normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);
        // //ת����8bitͼ����ʾ
        // cv::convertScaleAbs(imageGamma, imageGamma);
        // cv::imshow("٤���任ͼ����ǿЧ��", imageGamma);
        // cv::imshow("image_clone3", image_clone);
        // cv::waitKey();

        extract_pc_feature_6(filtered, edges);
        pc_feature.push_back(edges);
        // cv::waitKey(0);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_noground(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered2(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered3(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered4(new pcl::PointCloud<pcl::PointXYZI>);
        //
        //
        //        pcl::PassThrough<pcl::PointXYZI> pass_filter;
        //        pass_filter.setInputCloud(pc);
        //        pass_filter.setFilterFieldName("x");
        //        pass_filter.setFilterLimitsNegative(false);
        //        pass_filter.setFilterLimits(x_min, x_max);
        //        pass_filter.filter(*filtered);
        //
        ////        pcl::RadiusOutlierRemoval<pcl::PointXYZI> r_filter;
        ////        r_filter.setInputCloud(filtered);
        ////        r_filter.setRadiusSearch(search_r);
        ////        r_filter.setMinNeighborsInRadius(search_num);
        ////        r_filter.filter(*filtered2);
        ////
        ////        pass_filter.setFilterLimits(x_max, FLT_MAX);
        ////        pass_filter.filter(*filtered);
        ////        r_filter.setInputCloud(filtered);
        ////        r_filter.setRadiusSearch(search_r2);
        ////        r_filter.setMinNeighborsInRadius(search_num2);
        ////        r_filter.filter(*filtered3);
        ////
        ////        *filtered4 = *filtered2 + *filtered3;
        //
        //        std::vector<int> indices;
        //        pcl::removeNaNFromPointCloud(*filtered, *filtered4, indices);
        //        std::vector<pcl::PointCloud<PointXYZIA> > point_rings;
        //        PointXYZIA point;
        //        point_rings.resize(rings);
        //
        //
        //        for (int i = 0; i < filtered4->size(); i++) {
        //
        //            PointXYZIA point;
        //            int ring_id;
        //            float xiebian, angle;
        //            point.x = filtered4->points[i].x;
        //            point.y = filtered4->points[i].y;
        //            point.z = filtered4->points[i].z;
        //            point.intensity = filtered4->points[i].intensity;
        //            xiebian = std::sqrt(std::pow(point.y, 2) + std::pow(point.x, 2));
        //            point.cosangle = -(point.y / xiebian);
        //            point.distance = std::sqrt(std::pow(point.y, 2) + std::pow(point.x, 2) + std::pow(point.z, 2));
        //            angle = std::atan(point.z / xiebian);
        //            ring_id = int(((angle * 180 / M_PI) - lowerBound) * factor + 0.5);
        //            if (ring_id >= rings || ring_id < 0) {
        //                //std::cout << "\033[33mWarning: one point cannot find a ring!\033[0m" << std::endl;
        //                continue;
        //            }
        //
        //            { point_rings[ring_id].push_back(point); }
        //        }
        //
        //
        ////extract feature
        //        pcl::PointXYZI p;
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr edges(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered5(new pcl::PointCloud<pcl::PointXYZI>);
        //        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> edge_filtered;
        //        edge_filtered.resize(rings);
        //        int edge1 = 0;
        //        int edge2 = 0;
        //
        ////#pragma omp parallel for
        //        for (int i = 0; i < rings; i++) {
        //            if (debug) {
        //                if (omp_in_parallel()) {
        //                    std::cout << "in parallel" << std::endl;
        //                    std::cout << omp_get_num_threads() << " threads" << std::endl;
        //                } else {
        //                    std::cout << "not in parallel" << std::endl;
        //                }
        //            }
        //            pcl::PointCloud<pcl::PointXYZI>::Ptr haha(new pcl::PointCloud<pcl::PointXYZI>);
        //            edge_filtered[i] = haha;
        //            if (point_rings[i].size() >= 3) {
        //                pcl::PointXYZI p;
        //                sort(point_rings[i].points.begin(), point_rings[i].points.end(), pointcmp);
        //

        //                for (int j = 1; j < point_rings[i].size() - 1; j++) {
        //                    if (point_rings[i].points[j - 1].cosangle > point_rings[i].points[j].cosangle) {
        //                        std::cout << "\033[31mcountScore Error: Sort error!\033[0m" << std::endl;
        //                        std::exit(0);
        //                    }
        //                    //intensity = score
        //                    if ((point_rings[i].points[j - 1].distance - point_rings[i].points[j].distance) > dis_threshold ||
        //                        (point_rings[i].points[j + 1].distance - point_rings[i].points[j].distance) > dis_threshold) {
        //                        p.x = point_rings[i].points[j].x;
        //                        p.y = point_rings[i].points[j].y;
        //                        p.z = point_rings[i].points[j].z;
        //                        //p.intensity = point_rings[i].points[j-1].intensity;
        //                        p.intensity = (point_rings[i].points[j - 1].distance - point_rings[i].points[j].distance) >
        //                                      (point_rings[i].points[j + 1].distance - point_rings[i].points[j].distance) ? (
        //                                              point_rings[i].points[j - 1].distance - point_rings[i].points[j].distance)
        //                                                                                                                  : (
        //                                              point_rings[i].points[j + 1].distance -
        //                                              point_rings[i].points[j].distance);
        //                        haha->points.push_back(p);
        //                    } else {
        //                        float aa = std::acos(point_rings[i].points[j - 1].cosangle);
        //                        float bb = std::acos(point_rings[i].points[j].cosangle);
        //                        float cc = std::acos(point_rings[i].points[j + 1].cosangle);
        //                        float dis_angle = (aa - bb) > (bb - cc) ? (aa - bb) * 180 / M_PI : (bb - cc) * 180 / M_PI;
        //                        if (dis_angle > angle_threshold) {
        //                            p.x = point_rings[i].points[j].x;
        //                            p.y = point_rings[i].points[j].y;
        //                            p.z = point_rings[i].points[j].z;
        //                            //p.intensity = point_rings[i].points[j-1].intensity;
        //                            p.intensity = (aa - bb) * 180 / M_PI;
        //                            haha->points.push_back(p);
        //                            //std::cout << dis_angle << std::endl;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //
        //
        //        for (int i = 0; i < rings; i++) {
        //            edge1 += edge_filtered[i]->size();
        //            (*filtered5) += (*edge_filtered[i]);
        //        }
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr edges2(new pcl::PointCloud<pcl::PointXYZI>);
        //        std::cout << edge1 << ", " << edge2 << std::endl;
        ////        r_filter.setInputCloud(filtered5);
        ////        r_filter.setRadiusSearch(search_r3);
        ////        r_filter.setMinNeighborsInRadius(search_num3);
        ////        r_filter.filter(*edges2);
        //        float max_distance = config["max_distance"].as<float>();
        //        pass_filter.setInputCloud(filtered5);
        //        pass_filter.setFilterFieldName("x");
        //        pass_filter.setFilterLimitsNegative(false);
        //        pass_filter.setFilterLimits(0.00, max_distance);
        //        pass_filter.filter(*edges);
        //        pc_feature.push_back(edges);

        //extract image edges
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat raw_image, gray_image, gray_image_filter, hsv_image, edge_distance_image, edge_distance_image2, edge_distance_image3;
        cv::GaussianBlur(image, gray_image, cv::Size(5, 5), 2, 2); // 2, 2

        // cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
        if (gray_image.channels() == 1)
        {
            gray_image.copyTo(gray_image);
        }
        else
        {
            cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
        }
        cv::Mat image_edge;
        cv::Canny(gray_image, image_edge, canny_threshold_mini, canny_threshold_max);

        // cv::Sobel(gray_image, image_edge, -1, 1, 0);
        // cv::imshow("Canny", image_edge);
        // cv::waitKey(0);
        std::string i_string = std::to_string(i);

        std::string canny_string = "../data" + std::to_string(frame_cnt) + "/result/canny" + i_string + ".png";
        // cv::imwrite(canny_string, image_edge);

        cv::Mat image_edge_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));
        for (int i = 0; i < image_edge.cols; i++)
        {
            for (int j = 0; j < image_edge.rows; j++)
            {
                if (i >= 5 && j >= 5 && i <= (image_edge.cols - 5) && j <= (image_edge.rows - 5))
                {
                    float cen = (int)image_edge.at<uchar>(j, i);
                    float up = (int)image_edge.at<uchar>(j - 1, i);
                    float dw = (int)image_edge.at<uchar>(j + 1, i);
                    float lt = (int)image_edge.at<uchar>(j, i - 1);
                    float rt = (int)image_edge.at<uchar>(j, i + 1);
                    float lu = (int)image_edge.at<uchar>(j - 1, i - 1);
                    float ru = (int)image_edge.at<uchar>(j - 1, i + 1);
                    float ld = (int)image_edge.at<uchar>(j + 1, i - 1);
                    float rd = (int)image_edge.at<uchar>(j + 1, i + 1);
                    float uu = (int)image_edge.at<uchar>(j - 2, i);
                    float dd = (int)image_edge.at<uchar>(j + 2, i);
                    float luu = (int)image_edge.at<uchar>(j - 2, i - 1);
                    float ruu = (int)image_edge.at<uchar>(j - 2, i + 1);
                    float ldd = (int)image_edge.at<uchar>(j + 2, i - 1);
                    float rdd = (int)image_edge.at<uchar>(j + 2, i + 1);
                    float lluu = (int)image_edge.at<uchar>(j - 2, i - 2);
                    float rruu = (int)image_edge.at<uchar>(j - 2, i + 2);
                    float llu = (int)image_edge.at<uchar>(j - 1, i - 2);
                    float rru = (int)image_edge.at<uchar>(j - 1, i + 2);
                    float ll = (int)image_edge.at<uchar>(j, i - 2);
                    float rr = (int)image_edge.at<uchar>(j, i + 2);
                    float lld = (int)image_edge.at<uchar>(j + 1, i - 2);
                    float rrd = (int)image_edge.at<uchar>(j + 1, i + 2);
                    float lldd = (int)image_edge.at<uchar>(j + 2, i - 2);
                    float rrdd = (int)image_edge.at<uchar>(j + 2, i + 2);

                    // ��ֱ����
                    /*// if((cen==255||lu==255||up==255||ru==255||ld==255||dw==255||rd==255||luu==255||uu==255||ruu==255||
                    //     ldd==255||dd==255||rdd==255)
                    //     // && lluu==0&&llu==0&&ll==0&&lld==0&&lldd==0 && rruu==0&&rru==0&&rr==0&&rrd==0&&rrdd==0
                    //     )
                    // {
                    //     image_edge_copy.at<uchar>(j, i) = (int) image_edge.at<uchar>(j, i);
                    // } */
                    if (cen == 255 && (lu == 255 || up == 255 || ru == 255 || ld == 255 || dw == 255 || rd == 255) && lluu == 0 && llu == 0 && ll == 0 && lld == 0 && lldd == 0 && rruu == 0 && rru == 0 && rr == 0 && rrd == 0 && rrdd == 0)
                    {
                        image_edge_copy.at<uchar>(j, i) = (int)image_edge.at<uchar>(j, i);
                    }

                    // ˮƽ����
                    // if((cen==255||lu==255||lt==255||ld==255||llu==255||ll==255||lld==255||rt==255||ru==255||rd==255||
                    //     rru==255||rr==255||rrd==255)
                    //     && lluu==0&&luu==0&&uu==0&&ruu==0&&rruu==0 && lldd==0&&ldd==0&&dd==0&&rdd==0&&rrdd==0
                    //     )
                    // {
                    //     image_edge_copy.at<uchar>(j, i) = (int) image_edge.at<uchar>(j, i);
                    // }
                }
            }
        }
        // cv::imshow("canny_filter", image_edge_copy);
        // cv::waitKey(0);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter.png", image_edge_copy);

        cv::Mat image_edge_copy_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));
        for (int i = 0; i < image_edge_copy.cols; i++)
        {
            for (int j = 0; j < image_edge_copy.rows; j++)
            {
                if (i >= 5 && j >= 5 && i <= (image_edge_copy.cols - 5) && j <= (image_edge_copy.rows - 5))
                {
                    float cen = (int)image_edge_copy.at<uchar>(j, i);
                    float up = (int)image_edge_copy.at<uchar>(j - 1, i);
                    float dw = (int)image_edge_copy.at<uchar>(j + 1, i);
                    float lt = (int)image_edge_copy.at<uchar>(j, i - 1);
                    float rt = (int)image_edge_copy.at<uchar>(j, i + 1);
                    float lu = (int)image_edge_copy.at<uchar>(j - 1, i - 1);
                    float ru = (int)image_edge_copy.at<uchar>(j - 1, i + 1);
                    float ld = (int)image_edge_copy.at<uchar>(j + 1, i - 1);
                    float rd = (int)image_edge_copy.at<uchar>(j + 1, i + 1);

                    if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lt == 0 && rt == 0 && ld == 0 && rd == 0)
                    {
                        image_edge_copy_copy.at<uchar>(j, i) = 0;
                    }
                    else
                    {
                        image_edge_copy_copy.at<uchar>(j, i) = (int)image_edge_copy.at<uchar>(j, i);
                    }
                }
            }
        }
        // cv::imshow("canny_filter_filter", image_edge_copy_copy);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter_filter.png", image_edge_copy_copy);

        cv::Mat image_edge_copy_copy_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));
        for (int i = 0; i < image_edge_copy_copy.cols; i++)
        {
            for (int j = 0; j < image_edge_copy_copy.rows; j++)
            {
                if (i >= 5 && j >= 5 && i <= (image_edge_copy_copy.cols - 5) && j <= (image_edge_copy_copy.rows - 5))
                {
                    float cen = (int)image_edge_copy_copy.at<uchar>(j, i);
                    float up = (int)image_edge_copy_copy.at<uchar>(j - 1, i);
                    float dw = (int)image_edge_copy_copy.at<uchar>(j + 1, i);
                    float lt = (int)image_edge_copy_copy.at<uchar>(j, i - 1);
                    float rt = (int)image_edge_copy_copy.at<uchar>(j, i + 1);
                    float lu = (int)image_edge_copy_copy.at<uchar>(j - 1, i - 1);
                    float ru = (int)image_edge_copy_copy.at<uchar>(j - 1, i + 1);
                    float ld = (int)image_edge_copy_copy.at<uchar>(j + 1, i - 1);
                    float rd = (int)image_edge_copy_copy.at<uchar>(j + 1, i + 1);

                    if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lt == 0 && rt == 0 && ld == 0 && rd == 0)
                    {
                        image_edge_copy_copy_copy.at<uchar>(j, i) = 0;
                    }
                    else
                    {
                        image_edge_copy_copy_copy.at<uchar>(j, i) = (int)image_edge_copy_copy.at<uchar>(j, i);
                    }
                }
            }
        }
        // cv::imshow("canny_filter_filter_filter", image_edge_copy_copy_copy);
        // cv::waitKey(0);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter_filter_filter.png", image_edge_copy_copy_copy);

        // cv::Mat grad_x, abs_grad_x;
        // cv::Sobel(image_edge, grad_x, CV_16S, 1, 0, 1, 1, 1, cv::BORDER_DEFAULT);
        // cv::convertScaleAbs(grad_x, abs_grad_x);
        // cv::imshow("sobel", abs_grad_x);
        //  cv::waitKey(0);
        // std::vector<cv::Vec4i> Lines;
        // cv::HoughLinesP(image_edge, Lines, 1, CV_PI/180, 10, 10, 20);
        // for(int i = 0; i < Lines.size(); i++)
        // {
        //     if(abs(Lines[i][0]-Lines[i][2]) > 5) continue;
        //     cv::line(image, cv::Point(Lines[i][0], Lines[i][1]), cv::Point(Lines[i][2], Lines[i][3]), cv::Scalar(0, 0, 255), 2, 8);
        // }
        // cv::imshow("hough", image);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/hough.png", image);
        // cv::waitKey(0);
        // cv::imshow("image_edge_copy_copy_copy", image_edge_copy_copy_copy);
        // cv::waitKey(0);
        if (config["filter_edge_features"].as<bool>())
        {
            cv::bitwise_not(image_edge_copy_copy_copy, image_edge);
        }
        else
        {
            cv::bitwise_not(image_edge, image_edge);
        }

        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/image_edge.png", image_edge);
        // std::cout<<"imgage_edge type " << image_edge.type() << image_edge.channels() << std::endl;
        cv::distanceTransform(image_edge, edge_distance_image, CV_DIST_C, 3); //Method1
        // cv::distanceTransform(image_edge, edge_distance_image, CV_DIST_L2, 5);// Method2
        // cv::imshow("image_edge", image_edge);
        // cv::imshow("edge_distance_image", edge_distance_image);
        cv::normalize(edge_distance_image, edge_distance_image2, normalize_config, 0, cv::NORM_INF, 1); // Method1
                                                                                                        // std::cout << "img " << image_edge_copy_copy_copy.rows << " " << image_edge_copy_copy_copy.cols <<" " << image_edge.rows << " "
                                                                                                        // << image_edge.cols << " " << edge_distance_image.rows << " " << edge_distance_image.cols << " " <<
                                                                                                        // " " << edge_distance_image2.rows << " " << edge_distance_image2.cols << std::endl;
                                                                                                        // std::cout << "channels image_edge edge_distance_image edge_distance_image2 gray_image = " << image_edge.channels() << " " <<
                                                                                                        // edge_distance_image.channels() << " " << edge_distance_image2.channels() << std::endl;

        // cv::GaussianBlur(edge_distance_image, edge_distance_image2, cv::Size(3, 3), 1, 1);// 2, 2
        // cv::normalize(edge_distance_image, edge_distance_image2, 0, 5., NORM_MINMAX); // Method2
        // cv::imshow("edge_distance_image2", edge_distance_image2);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/edge_distance_image.png", edge_distance_image);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/edge_distance_image2.png", edge_distance_image2);
        // cv::waitKey(0);
        // std::cout<<"edge distance imag2 00 = " << (int) edge_distance_image2.at<uchar>(0, 0) <<std::endl;
        // std::cout<<"edge_distance_image2 = "<<edge_distance_image2<<std::endl;

//        cv::findContours(image_edge, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
//        cv::drawContours(image, contours, -1, cv::Scalar(0,0,255), 1, 8);
//      cv::imshow("test", image);
#pragma omp parallel for
        for (int i = 0; i < edge_distance_image2.cols; i++)
        {
            for (int j = 0; j < edge_distance_image2.rows; j++)
            {
                if (edge_distance_image2.at<uchar>(j, i) < 0 || edge_distance_image2.at<uchar>(j, i) > 255)
                {
                    std::cout << "error" << std::endl;
                    exit(0);
                }
                gray_image.at<uchar>(j, i) = 255 - (int)edge_distance_image2.at<uchar>(j, i);
                // std::cout << (int)gray_image.at<uchar>(j, i) << std::endl;
            }
        }
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/gray_image.png", gray_image);
        gray_image_vec.push_back(gray_image);
        // std::cout<<"gray_image 00 = " << (int) gray_image.at<uchar>(0, 0) <<std::endl;
        gray_image_filter = gray_image.clone();
        bool white_block;
        for (int i = 0; i < gray_image.cols; i++)
        {
            for (int j = 0; j < gray_image.rows; j++)
            {
                if (gray_image.at<uchar>(j, i) < 0 || gray_image.at<uchar>(j, i) > 255)
                {
                    std::cout << "error" << std::endl;
                    exit(0);
                }
                white_block = true;
                // std::cout<<"0"<<std::endl;
                if (i >= 10 && j >= 10 && i <= (gray_image.cols - 10) && j <= (gray_image.rows - 10))
                {
                    // std::cout<<"1"<<std::endl;
                    int img_cnt = 0;
                    int gray_pixel = 0;
                    for (int m = -5; m <= 5; m++)
                    {
                        for (int n = -5; n <= 5; n++)
                        {
                            // std::cout<<"2"<<std::endl;
                            img_cnt++;
                            gray_pixel += (int)gray_image.at<uchar>(j + m, i + n);
                            // if(abs((int)gray_image.at<uchar>(j+m, i+n)-255)>5){
                            //     // std::cout<<"3"<<std::endl;
                            //     white_block = false;
                            // }
                        }
                    }
                    gray_pixel = gray_pixel / img_cnt;
                    if (gray_pixel > 200)
                    {
                        // for(int m = -3; m <= 3; m++){
                        // for(int n = -3; n <= 3; n++){
                        // gray_image_filter.at<uchar>(j, i) = 127;
                        // }
                        // }
                    }
                }
                // std::cout<<"4"<<std::endl;
                // if(white_block){
                //     gray_image_filter.at<uchar>(j, i) = 0;
                // }
                // std::cout<<"5"<<std::endl;
                //std::cout << (int)edge_distance_image2.at<uchar>(j, i) << std::endl;
            }
        }
        // cv::imshow("gray_image", gray_image);
        // cv::imshow("gray_image_filter", gray_image_filter);
        // cv::waitKey(0);
        std::ostringstream str;
        str << i;
        // std::string image_gray = result_file[i].substr(0, result_file[i].length()-4);
        // std::string image_gray = "/home/zh/code/useful_tools/auto_calibration/data/result/image_gray" +  str.str() + ".png";
        // image_gray = "edge" + image_gray + ".png";
        // cv::imwrite(image_gray, gray_image_filter);

        // distance_image.push_back(gray_image);

        if (config["edge_features"].as<bool>()) //��������ȡ�������Ϸ�д�ļ�ʮ�����к����Ǳ�����
        {
            distance_image.push_back(gray_image_filter);
        }
        else if (config["line_features"].as<bool>()) //��������ȡ����
        {
            cv::Mat outimg, outimg_thin;
            std::vector<cv::line_descriptor::KeyLine> keylines, keylines2;
            extract_image_feature(image, image, keylines, keylines2, outimg, outimg_thin);
            distance_image.push_back(outimg); //outimg�ǵ�ͨ��Ҫ��Ϊ��ͨ��
            distance_image_thin.push_back(outimg_thin);
        }
    }
    std::cout << "End extract feature" << std::endl;
}

// 过滤多余点云(无法投影)
void AutoCalib::filterPCwithIMG(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, 
            const cv::Mat distance_image,
            Eigen::Matrix4f RT, Eigen::Matrix3f camera_param,
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered)
{

    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    int edge_size = pc_feature->size();
    float one_score = 0;
    int points_num = 0;

    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1)) //�ȹ��˵���ת������ͼ���ϵĵ�
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        if(distance_image.at<uchar>(y, x) == 0) continue;

        points_num++;

        pc_feature_filtered->push_back(r);
    }

}


//�������??
float AutoCalib::countScore(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                            Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{
    float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1)) //�ȹ��˵���ת������ͼ���ϵĵ�
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        //one_score += r.intensity * (int)distance_image[i].at<uchar>(y, x);

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//�����һ�����ص����150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;//��Χ���������ص����??150
        points_num++;

        // if((int) distance_image.at<uchar>(y, x) < 150)continue;//129

        double pt_dis = pow(r.x * r.x + r.y * r.y + r.z * r.z, double(1.0 / 2.0));
        //std::cout << r.x << "  " << r.y << "   " << r.z << "  " << pt_dis << std::endl;
        if (config["add_dis_weight"].as<bool>())
        {
            // one_score +=  (distance_image.at<uchar>(y, x) * sqrt(pc_feature->points[j].intensity));
            if (abs(r.intensity - 0.1) < 0.2)
            {
                one_score += (distance_image.at<uchar>(y, x) / pt_dis * 2) * 1.2; //x y�ǵ���ͶӰת������ͼ���ϵ���������
            }
            else
            {
                one_score += (distance_image.at<uchar>(y, x) / pt_dis * 2);
            }
        }
        else
        {
            one_score += distance_image.at<uchar>(y, x);
        }
    }
    // score = one_score;// / (float)data_num;
    score = one_score / 255.0 / points_num;
    if (config["many_or_one"].as<int>() == 2)
    {
        std::cout << "has " << points_num << std::endl;
    }

    return score;
}

float AutoCalib::countConfidence(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{
    float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    float points_whiter_200 = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 1 || x > (distance_image.cols - 2) || y < 1 || y > (distance_image.rows - 2))
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        //one_score += r.intensity * (int)distance_image[i].at<uchar>(y, x);

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//�����һ�����ص����150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;//��Χ���������ص����??150
        points_num++;

        float white;
        if(config["conf_pt"].as<bool>()){
            // Eigen::Matrix3f pattern;
            // pattern << 1, 2, 1,
            //                         2, 4, 2,
            //                         1, 2, 1;
            white = distance_image.at<uchar>(y-1, x-1)+distance_image.at<uchar>(y, x-1)*2+distance_image.at<uchar>(y+1, x-1)
                    + distance_image.at<uchar>(y-1, x)*2+distance_image.at<uchar>(y, x)*4+distance_image.at<uchar>(y+1, x)*2
                    + distance_image.at<uchar>(y-1, x+1)+distance_image.at<uchar>(y, x+1)*2+distance_image.at<uchar>(y+1, x+1);
            white /= (1+2+1+2+4+2+1+2+1);
        }
        else white = distance_image.at<uchar>(y, x);
        if (white > 130)
        {
            points_whiter_200++;
        }
    }

    score = points_whiter_200 / points_num;

    return score;
}

bool AutoCalib::isWhiteEnough(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                              Eigen::Matrix4f RT, Eigen::Matrix3f camera_param, bool fine_result, float &score)
{
    // float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    float points_num = 0;
    float points_whiter_200 = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1))
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        //one_score += r.intensity * (int)distance_image[i].at<uchar>(y, x);

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//�����һ�����ص����150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;//��Χ���������ص����??150
        points_num++;
        if (distance_image.at<uchar>(y, x) > 130)
        {
            points_whiter_200++;
        }

        // if((int) distance_image.at<uchar>(y, x) < 150)continue;//129
    }
    // std::cout << "random sample ========= " << points_whiter_200 / points_num << std::endl;
    score = points_whiter_200 / points_num;
    if (fine_result)
    {
        if (score > 0.6) //60%�ĵ����ڰ�ɫ���򣬷�����
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        if (score > 0.92)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

void AutoCalib::filterUnusedPoiintCloudFeature(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                                               Eigen::Matrix4f RT, Eigen::Matrix3f camera_param, pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered)
{
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    int edge_size = pc_feature->size();
    int one_score = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1)) //ͶӰ��ͼ������ĵ�??������
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        points_num++;

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//�����һ�����ص����150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;
        pc_feature_filtered->push_back(r);
    }
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> AutoCalib::get_in_pcs() //������node��30������
{
    return in_pcs;
}

std::vector <Eigen::Matrix4f>T_v_vec;
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>T_v_infer_cloud;
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> Message_filter_cloud;
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> Real_truth_cloud;
std::vector <float>score_count_cloud(6,0);
bool flag_once = true;

std::vector<cv::Mat> AutoCalib::get_in_images()
{
    return in_images;
}
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> AutoCalib::get_in_pcs_feature()
{
    return in_pcs_feature;
}

std::vector<cv::Mat> AutoCalib::get_in_images_feature()
{
    return in_images_feature;
}
Eigen::Matrix4f AutoCalib::get_in_pcs_current_guess()
{
    std::vector<float> ext = config["T_frame2frame0_pcs"]["data"].as<std::vector<float>>();
    Eigen::Matrix4f T_frame2frame0_pcs;
    assert((int)ext.size() == 16);
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            T_frame2frame0_pcs(row, col) = ext[row * 4 + col];
        }
    }
    return T_frame2frame0_pcs;
}
Eigen::Matrix4f AutoCalib::get_in_images_current_guess()
{
    std::vector<float> ext = config["T_frame2frame0_images"]["data"].as<std::vector<float>>();
    Eigen::Matrix4f T_frame2frame0_images;
    assert((int)ext.size() == 16);
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            T_frame2frame0_images(row, col) = ext[row * 4 + col];
        }
    }
    return T_frame2frame0_images;
}
Eigen::Matrix3f AutoCalib::get_in_k()
{
    return T_cam2image;
}
std::vector<Eigen::Matrix4f> AutoCalib::get_in_calibrated_result_vec() //�õ��궨�ĳ�ֵ
{
    return calibrated_result_vec;
}
bool AutoCalib::get_in_overlap() //�Ƿ����ص�������
{
    return overlap;
}
bool AutoCalib::get_in_add_dis_weight()
{
    return add_dis_weight;
}
Sophus::SE3d AutoCalib::toSE3d(Eigen::Matrix4f &T)
{
    Eigen::Matrix3d R;
    R(0, 0) = T(0, 0);
    R(0, 1) = T(0, 1);
    R(0, 2) = T(0, 2);
    R(1, 0) = T(1, 0);
    R(1, 1) = T(1, 1);
    R(1, 2) = T(1, 2);
    R(2, 0) = T(2, 0);
    R(2, 1) = T(2, 1);
    R(2, 2) = T(2, 2);
    Eigen::Quaterniond q(R);

    Eigen::Vector3d t(T(0, 3), T(1, 3), T(2, 3));
    Sophus::SE3d result(q, t);
    return result;
}
Eigen::Matrix4f AutoCalib::toMatrix4f(Eigen::Matrix4d s)
{
    Eigen::Matrix4f result;
    result(0, 0) = s(0, 0);
    result(0, 1) = s(0, 1);
    result(0, 2) = s(0, 2);
    result(0, 3) = s(0, 3);
    result(1, 0) = s(1, 0);
    result(1, 1) = s(1, 1);
    result(1, 2) = s(1, 2);
    result(1, 3) = s(1, 3);
    result(2, 0) = s(2, 0);
    result(2, 1) = s(2, 1);
    result(2, 2) = s(2, 2);
    result(2, 3) = s(2, 3);
    result(3, 0) = s(3, 0);
    result(3, 1) = s(3, 1);
    result(3, 2) = s(3, 2);
    result(3, 3) = s(3, 3);
    return result;
}

//主函数？
void AutoCalib::Run()  //3300-5200
{
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs, pcs_3frames, pc_features;
    std::vector<cv::Mat> images, distance_images, distance_images_thin;
    std::vector<float> oxts_vec;

    int dataNum = getData(config["txtname"].as<std::string>(), config["foldername"].as<std::string>(), pcs, images, oxts_vec);  
    if(config["sync"].as<bool>()){
        Get_message_filter_cloud(config["filtername"].as<std::string>(), config["filter_foldername"].as<std::string>(),Message_filter_cloud);
        Get_truth_cloud(config["truthname"].as<std::string>(), config["truth_foldername"].as<std::string>(),Real_truth_cloud);
    }
 
    in_images = images;
    in_pcs = pcs;

    if (pcs.size() < 2)
    {
        std::cout << "Too few point cloud frames" << std::endl;
    }

    if (config["down_sample"].as<bool>()) 
    {
        // Down Sample
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
        for (int i = 0; i < pcs.size(); ++i)
        {
            pcl::VoxelGrid<pcl::PointXYZI> pc_ds;
            pc_ds.setInputCloud(pcs[i]);
            pc_ds.setLeafSize(0.2f, 0.2f, 0.2f);
            pc_ds.filter(*pc_downsampled);
            pcl::copyPointCloud(*pc_downsampled, *pcs[i]);
        }
    }
    
    //融合三帧点云
    if (config["merge_frame"].as<bool>())
    {
        // Transform and Rotation Matrix, imu to velodyne.
        Eigen::Matrix4d T_imu2velo;
        T_imu2velo(0, 0) = 9.999976e-01;
        T_imu2velo(0, 1) = 7.553071e-04;
        T_imu2velo(0, 2) = -2.035826e-03;
        T_imu2velo(0, 3) = -8.086759e-01;
        T_imu2velo(1, 0) = -7.854027e-04;
        T_imu2velo(1, 1) = 9.998898e-01;
        T_imu2velo(1, 2) = -1.482298e-02;
        T_imu2velo(1, 3) = 3.195559e-01;
        T_imu2velo(2, 0) = 2.024406e-03;
        T_imu2velo(2, 1) = 1.482454e-02;
        T_imu2velo(2, 2) = 9.998881e-01;
        T_imu2velo(2, 3) = -7.997231e-01;
        T_imu2velo(3, 0) = 0;
        T_imu2velo(3, 1) = 0;
        T_imu2velo(3, 2) = 0;
        T_imu2velo(3, 3) = 1;

#define merged_frames 5
        // Merge 3 frames point cloud   ��������֡��һ֡�Ĵ��벿��
        for (int i = 0; i < pcs.size() - merged_frames - 1; ++i)
        {
            // for(int j = i+1; j <= i; ++j)
            for (int j = i + 1; j <= i + merged_frames - 1; ++j)
            {
                // Current and last eular angle from oxts file. (yaw pitch roll)
                Eigen::Vector3d oxts_eular_curr(oxts_vec[j * 30 + 5], oxts_vec[j * 30 + 4], oxts_vec[j * 30 + 3]);
                Eigen::Vector3d oxts_eular_last(oxts_vec[i * 30 + 5], oxts_vec[i * 30 + 4], oxts_vec[i * 30 + 3]);

                Eigen::Vector3d oxts_posi_curr, oxts_posi_last;
                double pi = 3.14159265358;
                double scale = std::cos(oxts_vec[j * 30] * pi / 180.0);
                double er = 6378137.0;
                double mx_curr = scale * oxts_vec[j * 30 + 1] * pi * er / 180.0;
                double my_curr = scale * er * std::log(std::tan((90.0 + oxts_vec[j * 30]) * pi / 360.0));
                double mx_last = scale * oxts_vec[i * 30 + 1] * pi * er / 180.0;
                double my_last = scale * er * std::log(std::tan((90.0 + oxts_vec[i * 30 + 0]) * pi / 360.0));

                oxts_posi_curr << mx_curr, my_curr, oxts_vec[j * 30 + 2];
                oxts_posi_last << mx_last, my_last, oxts_vec[i * 30 + 2];


                // Convert rotation from eular form to matrix form
                Eigen::Matrix3d R_imu_curr, R_imu_last, R_imu_last_counter, R_imu_curr_counter;


                R_imu_last = Eigen::AngleAxisd(oxts_eular_last[2], Eigen::Vector3d::UnitX()) *
                             Eigen::AngleAxisd(oxts_eular_last[1], Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(oxts_eular_last[0], Eigen::Vector3d::UnitZ());
                R_imu_curr = Eigen::AngleAxisd(oxts_eular_curr[2], Eigen::Vector3d::UnitX()) *
                             Eigen::AngleAxisd(oxts_eular_curr[1], Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(oxts_eular_curr[0], Eigen::Vector3d::UnitZ());


                Eigen::Matrix4d T_imu_curr, T_imu_last;

                T_imu_curr.block(0, 0, 3, 3) = R_imu_curr;
                T_imu_curr.block(0, 3, 3, 1) = oxts_posi_curr;
                T_imu_curr.block(3, 0, 1, 4) << 0, 0, 0, 1;

                T_imu_last.block(0, 0, 3, 3) = R_imu_last;
                T_imu_last.block(0, 3, 3, 1) = oxts_posi_last;
                T_imu_last.block(3, 0, 1, 4) << 0, 0, 0, 1;
                Eigen::Matrix4d T_imu_delt = T_imu_last.inverse() * T_imu_curr;

                Eigen::Matrix4d T_velo_delt = T_imu2velo * T_imu_last.inverse() * T_imu_curr * T_imu2velo.inverse();

                for (int m = 0; m < pcs[j]->points.size(); ++m) {
                    Eigen::Vector4d point_curr_frame(pcs[j]->points[m].x, pcs[j]->points[m].y, pcs[j]->points[m].z, 1);
                    Eigen::Vector4d point_last_frame = T_velo_delt * point_curr_frame;
                    pcl::PointXYZI point_temp;
                    point_temp.x = point_last_frame(0);
                    point_temp.y = point_last_frame(1);
                    point_temp.z = point_last_frame(2);
                    pcs[i]->push_back(point_temp);
                }
            }
        }
    }
    
    //提取特征的函数，传进去点云和图像，得到了点云和图像的线特征（LSD方法）
    extractFeature(pcs, images, pc_features, distance_images, distance_images_thin);
    
    in_pcs_feature = pc_features;
    in_images_feature = distance_images_thin;
    std::cout << "out extract feature" << std::endl;

            // pcl::visualization::PCLVisualizer viewer1("ttttt");
            // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> ttttt(pc_features[3], 0, 255, 0);  // GREEN
            // // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> filter_h(pcs[1], 0, 0, 255); // BLUE
            // // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> tgt_h(xxx, 255, 0, 0);  // RED

            // viewer1.setBackgroundColor(0,0,0);
            // viewer1.addPointCloud(pc_features[3], ttttt, "infer_cloud");
            // // viewer.addPointCloud(pcs[1], filter_h, "filter_cloud");
            // // viewer.addPointCloud(xxx, tgt_h, "truth_cloud");

            // while (!viewer1.wasStopped()){
            //     viewer1.spinOnce(100);
            //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            // }
    //读入扰动参数
    float bias_x, bias_y, bias_z;
    bias_x = config["bias_x"].as<float>();
    bias_y = config["bias_y"].as<float>();
    bias_z = config["bias_z"].as<float>();
    Eigen::AngleAxisf r_vx(M_PI * bias_x / 180, Eigen::Vector3f(1, 0, 0)); 
    Eigen::AngleAxisf r_vy(M_PI * bias_y / 180, Eigen::Vector3f(0, 1, 0));
    Eigen::AngleAxisf r_vz(M_PI * bias_z / 180, Eigen::Vector3f(0, 0, 1));
    Eigen::Matrix3f R_lidar2cam0_unbias = Eigen::Matrix3f::Identity();

    //读入雷达到相机0的变换矩阵
    std::vector<float> ext = config["R_lidar2cam0_unbias"]["data"].as<std::vector<float>>();
    assert((int)ext.size() == 9);
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            R_lidar2cam0_unbias(row, col) = ext[row * 3 + col];
        }
    }

    //变换矩阵转Euler角与四元数
    Eigen::Vector3f eulerAngle1 = R_lidar2cam0_unbias.eulerAngles(0, 1, 2);
    // std::cout<<"eulerAngle before  = "<<(180.0/3.14159*eulerAngle1)<<std::endl;
    Eigen::Quaternionf quaternion_before(R_lidar2cam0_unbias);
    // std::cout<<"quat before = "<<quaternion_before.w()<<" "<<quaternion_before.x()<<" "<<quaternion_before.y()<<" "<<quaternion_before.z()<<std::endl;

    //T_lidar2cam0_unbias lidar2cam0
    T_lidar2cam0_unbias(0, 0) = R_lidar2cam0_unbias(0, 0);
    T_lidar2cam0_unbias(0, 1) = R_lidar2cam0_unbias(0, 1);
    T_lidar2cam0_unbias(0, 2) = R_lidar2cam0_unbias(0, 2);
    T_lidar2cam0_unbias(1, 0) = R_lidar2cam0_unbias(1, 0);
    T_lidar2cam0_unbias(1, 1) = R_lidar2cam0_unbias(1, 1);
    T_lidar2cam0_unbias(1, 2) = R_lidar2cam0_unbias(1, 2);
    T_lidar2cam0_unbias(2, 0) = R_lidar2cam0_unbias(2, 0);
    T_lidar2cam0_unbias(2, 1) = R_lidar2cam0_unbias(2, 1);
    T_lidar2cam0_unbias(2, 2) = R_lidar2cam0_unbias(2, 2);
    T_lidar2cam0_unbias(0, 3) = config["t03"].as<float>();
    T_lidar2cam0_unbias(1, 3) = config["t13"].as<float>();
    T_lidar2cam0_unbias(2, 3) = config["t23"].as<float>();
    T_lidar2cam0_unbias(3, 0) = 0;
    T_lidar2cam0_unbias(3, 1) = 0;
    T_lidar2cam0_unbias(3, 2) = 0;
    T_lidar2cam0_unbias(3, 3) = 1;

    // 加偏差，只加了旋转偏差
    Eigen::Matrix3f R_lidar2cam0_bias = r_vx.matrix() * r_vy.matrix() * r_vz.matrix() * R_lidar2cam0_unbias; 
    
    Eigen::Vector3f eulerAngle2 = R_lidar2cam0_bias.eulerAngles(0, 1, 2);
    // std::cout<<"eulerAngle after = "<<(180.0/3.14159*eulerAngle2)<<std::endl;
    // std::cout<<"r_vx Angle = "<<(180.0/3.14159*(r_vx.matrix() * r_vy.matrix() * r_vz.matrix()).eulerAngles(0,1,2))<<std::endl;
    Eigen::Quaternionf quaternion_after(R_lidar2cam0_bias);
    // std::cout<<"quat after = "<<quaternion_after.w()<<" "<<quaternion_after.x()<<" "<<quaternion_after.y()<<" "<<quaternion_after.z()<<std::endl;

    T_lidar2cam0_bias(0, 0) = R_lidar2cam0_bias(0, 0);
    T_lidar2cam0_bias(0, 1) = R_lidar2cam0_bias(0, 1);
    T_lidar2cam0_bias(0, 2) = R_lidar2cam0_bias(0, 2);
    T_lidar2cam0_bias(1, 0) = R_lidar2cam0_bias(1, 0);
    T_lidar2cam0_bias(1, 1) = R_lidar2cam0_bias(1, 1);
    T_lidar2cam0_bias(1, 2) = R_lidar2cam0_bias(1, 2);
    T_lidar2cam0_bias(2, 0) = R_lidar2cam0_bias(2, 0);
    T_lidar2cam0_bias(2, 1) = R_lidar2cam0_bias(2, 1);
    T_lidar2cam0_bias(2, 2) = R_lidar2cam0_bias(2, 2);
    T_lidar2cam0_bias(0, 3) = config["t03"].as<float>();
    T_lidar2cam0_bias(1, 3) = config["t13"].as<float>();
    T_lidar2cam0_bias(2, 3) = config["t23"].as<float>();
    T_lidar2cam0_bias(3, 0) = 0;
    T_lidar2cam0_bias(3, 1) = 0;
    T_lidar2cam0_bias(3, 2) = 0;
    T_lidar2cam0_bias(3, 3) = 1;

    // 相机0到相机2
    std::vector<float> cam02cam2 = config["T_cam02cam2"]["data"].as<std::vector<float>>(); //��ǩ���õ�kitti���ݼ����õ��������ͼ�������Ҫ����������?���Ĳ���
    assert((int)cam02cam2.size() == 16);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            T_cam02cam2(row, col) = cam02cam2[row * 4 + col];
        }
    }
    T_lidar2cam2_bias = T_cam02cam2 * T_lidar2cam0_bias; 
    // 加位移偏差
    T_lidar2cam2_bias(0, 3) = T_lidar2cam2_bias(0, 3) + config["bias_t1"].as<float>();
    T_lidar2cam2_bias(1, 3) = T_lidar2cam2_bias(1, 3) + config["bias_t2"].as<float>();
    T_lidar2cam2_bias(2, 3) = T_lidar2cam2_bias(2, 3) + config["bias_t3"].as<float>();
    T_lidar2cam2_unbias = T_cam02cam2 * T_lidar2cam0_unbias;

    Eigen::Matrix3f R_lidar2cam2_bias = Eigen::Matrix3f::Identity();
    R_lidar2cam2_bias(0, 0) = T_lidar2cam2_bias(0, 0);
    R_lidar2cam2_bias(0, 1) = T_lidar2cam2_bias(0, 1);
    R_lidar2cam2_bias(0, 2) = T_lidar2cam2_bias(0, 2);
    R_lidar2cam2_bias(1, 0) = T_lidar2cam2_bias(1, 0);
    R_lidar2cam2_bias(1, 1) = T_lidar2cam2_bias(1, 1);
    R_lidar2cam2_bias(1, 2) = T_lidar2cam2_bias(1, 2);
    R_lidar2cam2_bias(2, 0) = T_lidar2cam2_bias(2, 0);
    R_lidar2cam2_bias(2, 1) = T_lidar2cam2_bias(2, 1);
    R_lidar2cam2_bias(2, 2) = T_lidar2cam2_bias(2, 2);

    //ros::init(argc, argv, "auto_calib");
    ros::Time::init();
    // ros::NodeHandle nh;

    //std::string score_evaluation_list_path = "../data" + std::to_string(frame_cnt) + "/result/calibrated_result"; //����calibrated_result���洢�ĵ�ַ
    //std::ofstream score_evaluation(calibrated_result_path, std::ios::app);  //calibrated_result���Ǹ�д����

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pc_features_local;
    std::vector<cv::Mat> distance_images_local, distance_images_local_thin;
    std::string calibrated_result_path = "../data" + std::to_string(frame_cnt) + "/result/calibrated_result"; //����calibrated_result���洢�ĵ�ַ
    std::ofstream calibrated_result(calibrated_result_path, std::ios::app);  //calibrated_result���Ǹ�д����
    Eigen::Matrix3f R_gt = T_lidar2cam2_unbias.block(0, 0, 3, 3);
    Eigen::Vector3f euler_ang_gt_zyx = R_gt.eulerAngles(0, 1, 2); 
    std::vector<Eigen::Vector3f> euler_ang_calibrated_zyx_vec, euler_ang_inv_delt_zyx_vec, euler_ang_inv_delt_xyz_vec,
        euler_ang_delt_zyx_vec, euler_ang_delt_zyx_vec_mean, euler_ang_delt_xyz_vec;
    std::vector<int> large_small_step;
    //加偏差前的变化（欧拉角）
    calibrated_result << "xyz = ";
    calibrated_result << euler_ang_gt_zyx[2] * 180.0 / M_PI << ",";
    calibrated_result << euler_ang_gt_zyx[1] * 180.0 / M_PI << ",";
    calibrated_result << euler_ang_gt_zyx[0] * 180.0 / M_PI;


    // 是否改用语义方法
    std::vector<cv::Mat> dstimgs, fullimgs;
    if(config["yy"].as<bool>()) {
        // 读语义数据
        std::vector<cv::Mat> other_images;
        Get_other_images(config["yyname"].as<std::string>(), config["yy_foldername"].as<std::string>(), other_images);
        std::vector<cv::Mat> other_images_blur(other_images), block_imgs(other_images), 
                                                dstimgs_fat, dstimgs_thin, testimgs;

        // 处理语义图像特征
        for(size_t i = 0; i < block_imgs.size(); ++i) {
            // distance transform for each image
            auto srcimg = block_imgs[i];
            cv::cvtColor(srcimg, srcimg, cv::COLOR_RGB2GRAY);

            for( auto i = srcimg.begin<uchar>(); i != srcimg.end<uchar>(); ++i) {
                if(*i == 0) continue;
                *i = 255;
            }

            cv::Mat bitmap, distimg, distimg_fat, distimg_thin, testimg;
            cv::threshold(srcimg, bitmap, 50, 255, cv::THRESH_BINARY);
            cv::distanceTransform(bitmap, distimg, 1, 3);
            cv::normalize(distimg, distimg_fat, config["norm_in_1"].as<int>(), 0, cv::NORM_INF);
            cv::normalize(distimg, distimg_thin, config["norm_in_2"].as<int>(), 0, cv::NORM_INF);
            cv::normalize(distimg, testimg, 0, 255, cv::NORM_MINMAX);
            distimg_thin.convertTo(distimg_thin, CV_8U);
            distimg_fat.convertTo(distimg_fat, CV_8U);
            testimg.convertTo(testimg, CV_8U);

            for( auto i = distimg_fat.begin<uchar>(); i != distimg_fat.end<uchar>(); ++i) {
                if(*i == 0) continue;
                *i = 255-*i;
            }
            for( auto i = distimg_thin.begin<uchar>(); i != distimg_thin.end<uchar>(); ++i) {
                if(*i == 0) continue;
                *i = 255-*i;
            }
            for( auto i = testimg.begin<uchar>(); i != testimg.end<uchar>(); ++i) {
                if(*i == 0) continue;
                *i = 255-*i;
            }

            cv::Mat notbitmap = ~bitmap;
            cv::Mat notdistimg, notdistimg_fat, notdistimg_thin;
            cv::distanceTransform(notbitmap, notdistimg, 1, 3);
            cv::normalize(notdistimg, notdistimg_fat, config["norm_out_1"].as<int>(), 0, cv::NORM_INF);
            cv::normalize(notdistimg, notdistimg_thin, config["norm_out_2"].as<int>(), 0, cv::NORM_INF);
            notdistimg_thin.convertTo(notdistimg_thin, CV_8U);
            notdistimg_fat.convertTo(notdistimg_fat, CV_8U);

            for( auto i = notdistimg_fat.begin<uchar>(); i != notdistimg_fat.end<uchar>(); ++i) {
                if(*i == 0) continue;
                *i = 255-*i;
            }
            for( auto i = notdistimg_thin.begin<uchar>(); i != notdistimg_thin.end<uchar>(); ++i) {
                if(*i == 0) continue;
                *i = 255-*i;
            }

            cv::Mat dstimg_fat, dstimg_thin;
            add(distimg_fat, notdistimg_fat, dstimg_fat);
            add(distimg_thin, notdistimg_thin, dstimg_thin);
            // cv::imshow("fat", dstimg_fat);
            // cv::waitKey(0);
            // cv::imshow("thin", dstimg_thin);
            // cv::waitKey(0);

            dstimgs.push_back(dstimg_thin);
            dstimgs_fat.push_back(dstimg_fat);
            dstimgs_thin.push_back(dstimg_thin);
            testimgs.push_back(testimg);
            
            // 剔除无效雷达点用到的加粗版bitmap
            int coresize = config["coresize"].as<int>();
            int coresigma = config["coresigma"].as<int>();
            
            std::vector<int> kvec(config["kernel"]["data"].as<std::vector<int>>());
            cv::Mat tmpimg(kvec);
            
            cv::Mat_<uchar> kern = tmpimg.reshape(0, config["kernel"]["rows"].as<int>());
            
            int loop =  config["kernel"]["loop"].as<int>();

            for(int i = 0; i < loop; ++i) {
                cv::filter2D(bitmap, bitmap, bitmap.depth(), kern);
            }

            // cv::imshow("bitmap", bitmap);
            // cv::waitKey(0);

            fullimgs.push_back(bitmap);
        }

        // cv::imshow("testmap", testimgs[2]);
        // cv::waitKey(0);
        // cv::imshow("thinmap", dstimgs_thin[2]);
        // cv::waitKey(0);
        // cv::imshow("fatmap", dstimgs_fat[2]);
        // cv::waitKey(0);

// #define blur
#ifdef blur
    other_images_blur.clear();
    
    int coresize = config["coresize"].as<int>();
    int coresigma = config["coresigma"].as<int>();
    
    std::vector<int> kvec(config["kernel"]["data"].as<std::vector<int>>());
    cv::Mat tmpimg(kvec);
    
    cv::Mat_<uchar> kern = tmpimg.reshape(0, config["kernel"]["rows"].as<int>());
    
    int loop =  config["kernel"]["loop"].as<int>();

    for( cv::Mat ig:other_images ) {
        cv::Mat img;
        ig.copyTo(img);
        cv::Mat dstimg;
        for(int i = 0; i < loop; ++i) {
            cv::filter2D(img, dstimg, img.depth(), kern);
            cv::threshold(dstimg, img, 130, 255, CV_THRESH_BINARY);
        }
        
        cv::Mat img_blur;
        cv::GaussianBlur(img, img_blur, cv::Size(coresize,coresize), coresigma, 0);
        other_images_blur.push_back(img_blur);
    }
#endif
        if(config["testimg"].as<bool>()) {
            for (int ppp = 0; ppp < pcs.size(); ppp++)
            {
                pc_features_local.push_back(pc_features[ppp]);
                distance_images_local.push_back(testimgs[ppp]);
                distance_images_local_thin.push_back(testimgs[ppp]);
            }
        }
        else{
            for (int ppp = 0; ppp < pcs.size(); ppp++)
            {
                pc_features_local.push_back(pc_features[ppp]);
                distance_images_local.push_back(dstimgs_fat[ppp]);
                distance_images_local_thin.push_back(dstimgs_thin[ppp]);
            }
        }
        
    }
    else{
        for (int ppp = 0; ppp < pcs.size(); ppp++)
        {
            pc_features_local.push_back(pc_features[ppp]);
            distance_images_local.push_back(distance_images[ppp]);
            distance_images_local_thin.push_back(distance_images_thin[ppp]);
        }
    }
    
    // 读入修正参数
    float iterate_ang_step_big[2];
    iterate_ang_step_big[0] = config["iterate_ang_step_big[0]"].as<float>(); 
    iterate_ang_step_big[1] = config["iterate_ang_step_big[1]"].as<float>();
    float iterate_tra_step_big[2];
    iterate_tra_step_big[0] = config["iterate_tra_step_big[0]"].as<float>();
    iterate_tra_step_big[1] = config["iterate_tra_step_big[1]"].as<float>();

    float iterate_ang_step_small[2];
    iterate_ang_step_small[0] = config["iterate_ang_step_small[0]"].as<float>();
    iterate_ang_step_small[1] = config["iterate_ang_step_small[1]"].as<float>();
    float iterate_tra_step_small[2];
    iterate_tra_step_small[0] = config["iterate_tra_step_small[0]"].as<float>();
    iterate_tra_step_small[1] = config["iterate_tra_step_small[1]"].as<float>();

    bool got_fine_result = false;
    std::vector<float> confidence_vec;
    //Eigen::Matrix4f T_lidar2cam2_bias_temp = T_lidar2cam2_bias; //T_lidar2cam2_bias
    pcl::PointCloud<pcl::PointXYZI>::Ptr dynamic_cloud(new pcl::PointCloud<pcl::PointXYZI>);
     // 计算惯导修正
     std::vector<Eigen::Matrix4f> T_vec;
    if(config["sync"].as<bool>()){
        for(int ppp = 0; ppp < pcs.size(); ppp++)
        {
            Eigen::AngleAxisf rotation_vector(-oxts_vec[(ppp) * 30 + 5] + oxts_vec[(ppp-1) * 30 + 5], Eigen::Vector3f(0, 0, 1));
            Eigen::Matrix3f rotation_matrix = rotation_vector.matrix();
            Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
            T.rotate(rotation_vector);
            T.pretranslate(Eigen::Vector3f(-oxts_vec[ppp * 30 + 8] * 0.1, -oxts_vec[ppp * 30 + 9] * 0.1, 0));
            T_vec.push_back(T.matrix());
        } 
    }

    //Kitti
    for (int ppp = 1; ppp < pcs.size(); ppp++)
    {
        // 加扰动
        if( ppp == 8 || ppp == 15) {
            T_lidar2cam2_bias.block<3,3>(0,0) = r_vx.matrix() * r_vy.matrix() * r_vz.matrix() * T_lidar2cam2_bias.block<3,3>(0,0);
            T_lidar2cam2_bias(0, 3) = T_lidar2cam2_bias(0, 3) + config["bias_t1"].as<float>();
            T_lidar2cam2_bias(1, 3) = T_lidar2cam2_bias(1, 3) + config["bias_t2"].as<float>();
            T_lidar2cam2_bias(2, 3) = T_lidar2cam2_bias(2, 3) + config["bias_t3"].as<float>();
        }
        if(config["sync"].as<bool>()){
            Eigen::Matrix4f TT = T_vec[ppp-1]*T_vec[ppp];
            
            // if(ppp%2 == 1) T_lidar2cam2_bias = T_lidar2cam2_unbias * TT;
            // else T_lidar2cam2_bias = T_lidar2cam2_unbias * T_vec[ppp];

            //T_lidar2cam2_bias = T_lidar2cam2_unbias;

            //去除动态物体
            // if(ppp%2 == 0) icp_erase(pc_features_local[ppp], pc_features_local[ppp - 2], TT, dynamic_cloud);
            // else pc_features_local[ppp] =  pc_features_local[ppp-1];
            
            icp_erase(pc_features_local[ppp], pc_features[ppp - 1], T_vec[ppp], dynamic_cloud);
        }
        
        // 若使用语义方法，过滤点云
        if(config["yy"].as<bool>()) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered(new pcl::PointCloud<pcl::PointXYZI>);
            
            filterPCwithIMG(pc_features_local[ppp], 
                fullimgs[ppp], T_lidar2cam2_bias, 
                T_cam2image, pc_feature_filtered);
            
            pc_features_local[ppp] = pc_feature_filtered;

        }

        cv::Mat image_before_optimize;
        cv::Mat image_unbias;
        project2image(pc_features_local[ppp], images[ppp], image_unbias, T_lidar2cam2_unbias, T_cam2image);
        cv::Mat image1;
        project2image(pc_features_local[ppp], distance_images_local[ppp], image1, T_lidar2cam2_bias, T_cam2image);
        cv::Mat image_bias;
        project2image(pc_features_local[ppp], images_withouthist[ppp], image_bias, T_lidar2cam2_bias, T_cam2image);
        cv::Mat image_true;
        project2image(pc_features_local[ppp], images[ppp], image_true, T_lidar2cam2_unbias, T_cam2image);
        cv::Mat before_on_image, before_on_semantic;
        project2image(pc_features_local[ppp], images[ppp], before_on_image, T_lidar2cam2_bias, T_cam2image);
        project2image(pc_features_local[ppp], distance_images_local_thin[ppp], before_on_semantic, T_lidar2cam2_bias, T_cam2image);
        cv::Mat fullimg;
        project2image(pc_features_local[ppp], fullimgs[ppp], fullimg, T_lidar2cam2_bias, T_cam2image);

        omp_lock_t mylock;
        omp_init_lock(&mylock);
        float starTime = omp_get_wtime();
        float tran_x, tran_y, tran_z;
        float rot_x_adjust, rot_y_adjust, rot_z_adjust;
        tran_x = T_lidar2cam2_bias(0, 3);
        tran_y = T_lidar2cam2_bias(1, 3);
        tran_z = T_lidar2cam2_bias(2, 3);
        rot_x_adjust = 0;
        rot_y_adjust = 0;
        rot_z_adjust = 0;
        float max_score = 0;
        int nn = 0;
        int num_ten_thous = 0;
        if (got_fine_result)
        {
            max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
        }
        else
        {
            max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
        }

        if (config["many_or_one"].as<int>() == 1)
        {
#pragma omp parallel for //��ǩ:�������Ż�
            for (int i = 0; i < 1000 * 125; i++)
            {
                int dr = i / 1000;       // 0-125
                int drx = dr / 25;       // 0-5
                int dry = (dr / 5) % 5;  // 0-5
                int drz = dr % 5;        // 0-5
                int dt = i % 1000;       // 0-1000
                int dx = (dt / 100);     // 0-10
                int dy = (dt / 10) % 10; // 0-10
                int dz = dt % 10;        // 0-10
                float score = 0;
                Eigen::Matrix4f T_lidar2cam2_bias_copy; // = T_lidar2cam2_bias;
                // std::cout << "(" << dx << ", " << dy << ", " << dz << ")" << std::endl;
                Eigen::AngleAxisf rot_dx(M_PI * (-0.2 + 0.1 * drx) / 180, Eigen::Vector3f(1, 0, 0));
                Eigen::AngleAxisf rot_dy(M_PI * (-0.2 + 0.1 * dry) / 180, Eigen::Vector3f(0, 1, 0));
                Eigen::AngleAxisf rot_dz(M_PI * (-0.2 + 0.1 * drz) / 180, Eigen::Vector3f(0, 0, 1));
                Eigen::Matrix3f R_lidar2cam2_bias_temp = Eigen::Matrix3f::Identity();
                R_lidar2cam2_bias_temp = rot_dx.matrix() * rot_dy.matrix() * rot_dz.matrix() * T_lidar2cam2_bias.block<3, 3>(0, 0);
                ;
                T_lidar2cam2_bias_copy(0, 0) = R_lidar2cam2_bias_temp(0, 0);
                T_lidar2cam2_bias_copy(0, 1) = R_lidar2cam2_bias_temp(0, 1);
                T_lidar2cam2_bias_copy(0, 2) = R_lidar2cam2_bias_temp(0, 2);
                T_lidar2cam2_bias_copy(1, 0) = R_lidar2cam2_bias_temp(1, 0);
                T_lidar2cam2_bias_copy(1, 1) = R_lidar2cam2_bias_temp(1, 1);
                T_lidar2cam2_bias_copy(1, 2) = R_lidar2cam2_bias_temp(1, 2);
                T_lidar2cam2_bias_copy(2, 0) = R_lidar2cam2_bias_temp(2, 0);
                T_lidar2cam2_bias_copy(2, 1) = R_lidar2cam2_bias_temp(2, 1);
                T_lidar2cam2_bias_copy(2, 2) = R_lidar2cam2_bias_temp(2, 2);
                T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3) - 0.1 + dx * 0.01;
                T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3) - 0.1 + dy * 0.01;
                T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3) - 0.1 + dz * 0.01;
                score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                omp_set_lock(&mylock);
                {
                    if (nn / 10000 >= num_ten_thous)
                    {
                        num_ten_thous++;
                        std::cout << "n: " << 10000 * (int)(nn / 10000) << std::endl;
                    }
                    nn++;
                    if (score > max_score)
                    {
                        max_score = score;
                        tran_x = T_lidar2cam2_bias_copy(0, 3);
                        tran_y = T_lidar2cam2_bias_copy(1, 3);
                        tran_z = T_lidar2cam2_bias_copy(2, 3);
                        rot_x_adjust = -0.2 + 0.1 * drx;
                        rot_y_adjust = -0.2 + 0.1 * dry;
                        rot_z_adjust = -0.2 + 0.1 * drz;
                    }
                }
                omp_unset_lock(&mylock);
            }
            std::cout << "image " << ppp << " :" << std::endl;
            std::cout << "Max_Score: " << max_score << std::endl;
            std::cout << "(tran_x, tran_y, tran_z) = " << tran_x << ", " << tran_y << ", " << tran_z << std::endl;
            std::cout << "(new_rx, new_ry, new_rz) = " << bias_x + rot_x_adjust << ", " << bias_y + rot_y_adjust << ", " << bias_z + rot_z_adjust << std::endl;
            std::cout << "Time: " << omp_get_wtime() - starTime << std::endl;
        }
        if (config["many_or_one"].as<int>() == 2)
        {
            max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            std::cout << "Max_Score: " << max_score << std::endl;
        }

        std::cout << "Kitti GoundTruth score   =  " << countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_unbias, T_cam2image) << std::endl;
        std::cout << "Before Calibration score =  " << countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image) << std::endl;
        bool flag = true;
        if (config["many_or_one"].as<int>() == 3)
        {
            //T_lidar2cam_top3, T_lidar2image is Matrix3*4 //lida2image=T*(T_cam02cam2)*T_cam2image

            T_lidar2cam_top3 = T_lidar2cam2_bias.topRows(3);
            T_lidar2image = T_cam2image * T_lidar2cam_top3;
            float better_cnt = 0;
            
            //搜索法开始

            for(int iii=0; iii<2 ;iii++){
                if(iii==0){
                    got_fine_result = false;
                    cout << "Start 1" << endl;
                    float white_score;
                    got_fine_result = isWhiteEnough(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, white_score);
                    if(got_fine_result){
                        cout << "The first loop use distance_thin" << endl;
                        iii = 1;
                    } 
                }
                else cout << "Start 2" << endl;

                if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
                {
                    if (got_fine_result)
                    {
                        max_score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                    }
                    else
                    {
                        max_score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                    }
                }
                else
                {
                    if (got_fine_result)
                    {
                        max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
                    }
                    else
                    {
                        max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                    }
                }

                // 大步长搜索法优化
                flag = true;
                while (flag)
                {
                    flag = false;
                    Eigen::Matrix4f T_lidar2cam2_bias_big = T_lidar2cam2_bias; 

                    #pragma omp parallel for num_threads(1)
                    //Both adjusting tran and angle
                    for (int i = 0; i < 27*27; i++) {
                        int dr = i / 27;      // 0-125
                        int drx = dr / 9;     // 0-5
                        int dry = (dr / 3) % 3; // 0-5
                        int drz = dr % 3;       // 0-5
                        int dt = i % 27;      // 0-1000
                        int dx = (dt / 9);    // 0-10
                        int dy = (dt / 3) % 3;// 0-10
                        int dz = dt % 3;       // 0-
                    
                    // Only update angle
                    // for (int i = 0; i < 27; i++)
                    // {
                    //     int drx = i / 9;       // 0-5
                    //     int dry = (i / 3) % 3; // 0-5
                    //     int drz = i % 3;       // 0-5

                        Eigen::Matrix4f T_lidar2cam2_bias_copy;                                                                                // = T_lidar2cam2_bias;
                        Eigen::AngleAxisf rot_dx(M_PI * (-iterate_ang_step_big[iii] + iterate_ang_step_big[iii] * drx) * config["rx_plus"].as<float>() / 180, Eigen::Vector3f(1, 0, 0)); //0.06
                        Eigen::AngleAxisf rot_dy(M_PI * (-iterate_ang_step_big[iii] + iterate_ang_step_big[iii] * dry) * config["ry_plus"].as<float>() / 180, Eigen::Vector3f(0, 1, 0));
                        Eigen::AngleAxisf rot_dz(M_PI * (-iterate_ang_step_big[iii] + iterate_ang_step_big[iii] * drz) * config["rz_plus"].as<float>() / 180, Eigen::Vector3f(0, 0, 1));
                        Eigen::Matrix3f R_lidar2cam2_bias_temp = Eigen::Matrix3f::Identity();
                        R_lidar2cam2_bias_temp = rot_dx.matrix() * rot_dy.matrix() * rot_dz.matrix() * T_lidar2cam2_bias.block<3, 3>(0, 0);

                        T_lidar2cam2_bias_copy(0, 0) = R_lidar2cam2_bias_temp(0, 0);
                        T_lidar2cam2_bias_copy(0, 1) = R_lidar2cam2_bias_temp(0, 1);
                        T_lidar2cam2_bias_copy(0, 2) = R_lidar2cam2_bias_temp(0, 2);
                        T_lidar2cam2_bias_copy(1, 0) = R_lidar2cam2_bias_temp(1, 0);
                        T_lidar2cam2_bias_copy(1, 1) = R_lidar2cam2_bias_temp(1, 1);
                        T_lidar2cam2_bias_copy(1, 2) = R_lidar2cam2_bias_temp(1, 2);
                        T_lidar2cam2_bias_copy(2, 0) = R_lidar2cam2_bias_temp(2, 0);
                        T_lidar2cam2_bias_copy(2, 1) = R_lidar2cam2_bias_temp(2, 1);
                        T_lidar2cam2_bias_copy(2, 2) = R_lidar2cam2_bias_temp(2, 2);
                        //Both adjusting tran and angle
                        T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3) + (- iterate_tra_step_big[iii] + dx * iterate_tra_step_big[iii]) * config["tx_plus"].as<float>();//0.002
                        T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3) + (- iterate_tra_step_big[iii] + dy * iterate_tra_step_big[iii]) * config["ty_plus"].as<float>();
                        T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3) + (- iterate_tra_step_big[iii] + dz * iterate_tra_step_big[iii]) * config["tz_plus"].as<float>();

                        //Only update angle
                        // T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3); // - iterate_tra_step_big + dx * iterate_tra_step_big;//0.002
                        // T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3); // - iterate_tra_step_big + dy * iterate_tra_step_big;
                        // T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3); // - iterate_tra_step_big + dz * iterate_tra_step_big;

                        float score = 0;
                        if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
                        {
                            if (got_fine_result)
                            {
                                score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                            }
                            else
                            {
                                score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                            }
                        }
                        else
                        {
                            // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                            if (got_fine_result)
                            {
                                score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                            }
                            else
                            {
                                score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                            }
                        }
                        omp_set_lock(&mylock); //最速下降
                        {
                            if (score > max_score)
                            {
                                flag = true;
                                // i = 0;
                                max_score = score;
                                T_lidar2cam2_bias_big(0, 0) = T_lidar2cam2_bias_copy(0, 0);
                                T_lidar2cam2_bias_big(0, 1) = T_lidar2cam2_bias_copy(0, 1);
                                T_lidar2cam2_bias_big(0, 2) = T_lidar2cam2_bias_copy(0, 2);
                                T_lidar2cam2_bias_big(0, 3) = T_lidar2cam2_bias_copy(0, 3);
                                T_lidar2cam2_bias_big(1, 0) = T_lidar2cam2_bias_copy(1, 0);
                                T_lidar2cam2_bias_big(1, 1) = T_lidar2cam2_bias_copy(1, 1);
                                T_lidar2cam2_bias_big(1, 2) = T_lidar2cam2_bias_copy(1, 2);
                                T_lidar2cam2_bias_big(1, 3) = T_lidar2cam2_bias_copy(1, 3);
                                T_lidar2cam2_bias_big(2, 0) = T_lidar2cam2_bias_copy(2, 0);
                                T_lidar2cam2_bias_big(2, 1) = T_lidar2cam2_bias_copy(2, 1);
                                T_lidar2cam2_bias_big(2, 2) = T_lidar2cam2_bias_copy(2, 2);
                                T_lidar2cam2_bias_big(2, 3) = T_lidar2cam2_bias_copy(2, 3);
                                T_lidar2cam2_bias_big(3, 0) = T_lidar2cam2_bias_copy(3, 0);
                                T_lidar2cam2_bias_big(3, 1) = T_lidar2cam2_bias_copy(3, 1);
                                T_lidar2cam2_bias_big(3, 2) = T_lidar2cam2_bias_copy(3, 2);
                                T_lidar2cam2_bias_big(3, 3) = T_lidar2cam2_bias_copy(3, 3);
                                better_cnt++;
                            }
                        }
                        omp_unset_lock(&mylock);
                    }
                    T_lidar2cam2_bias = T_lidar2cam2_bias_big;
                    std::cout << "better_cnt first = " << better_cnt << std::endl;

                    T_lidar2cam_top3 = T_lidar2cam2_bias.topRows(3);
                    T_lidar2image = T_cam2image * T_lidar2cam_top3;
                } //while loop end

                if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
                {
                    if (got_fine_result)
                    {
                        max_score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                    }
                    else
                    {
                        max_score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                    }
                }
                else
                {
                    // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                    if (got_fine_result)
                    {
                        max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
                    }
                    else
                    {
                        max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                    }
                }

                // 小步长搜索法优化
                flag = true;
                while (flag)
                {
                    flag = false;
                    Eigen::Matrix4f T_lidar2cam2_bias_small = T_lidar2cam2_bias;

                    #pragma omp parallel for num_threads(1)
                    //Both adjusting tran and angle
                    for (int i = 0; i < 27*27; i++) {
                        int dr = i / 27;      // 0-125
                        int drx = dr / 9 ;     // 0-5
                        int dry = (dr / 3) % 3; // 0-5
                        int drz = dr % 3;       // 0-5
                        int dt = i % 27;      // 0-1000
                        int dx = (dt / 9);    // 0-10
                        int dy = (dt / 3) % 3;// 0-10
                        int dz = dt % 3;       // 0-

                    // Only update angle
                    // for (int i = 0; i < 27; i++)
                    // {
                    //     int drx = i / 9;       // 0-5
                    //     int dry = (i / 3) % 3; // 0-5
                    //     int drz = i % 3;       // 0-5

                        Eigen::Matrix4f T_lidar2cam2_bias_copy;                                                                                    // = T_lidar2cam2_bias;
                        Eigen::AngleAxisf rot_dx(M_PI * (-iterate_ang_step_small[iii] + iterate_ang_step_small[iii] * drx) * config["rx_plus"].as<float>() / 180, Eigen::Vector3f(1, 0, 0)); //0.01
                        Eigen::AngleAxisf rot_dy(M_PI * (-iterate_ang_step_small[iii] + iterate_ang_step_small[iii] * dry) * config["ry_plus"].as<float>() / 180, Eigen::Vector3f(0, 1, 0));
                        Eigen::AngleAxisf rot_dz(M_PI * (-iterate_ang_step_small[iii] + iterate_ang_step_small[iii] * drz) * config["rz_plus"].as<float>() / 180, Eigen::Vector3f(0, 0, 1));
                        Eigen::Matrix3f R_lidar2cam2_bias_temp = Eigen::Matrix3f::Identity();
                        // R_lidar2cam2_bias_temp(0, 0)=T_lidar2cam2_bias_copy(0,0);
                        // R_lidar2cam2_bias_temp(0, 1)=T_lidar2cam2_bias_copy(0,1);
                        // R_lidar2cam2_bias_temp(0, ;2)=T_lidar2cam2_bias_copy(0,2);
                        // R_lidar2cam2_bias_temp(1, 0)=T_lidar2cam2_bias_copy(1,0);
                        // R_lidar2cam2_bias_temp(1, 1)=T_lidar2cam2_bias_copy(1,1);
                        // R_lidar2cam2_bias_temp(1, 2)=T_lidar2cam2_bias_copy(1,2);
                        // R_lidar2cam2_bias_temp(2, 0)=T_lidar2cam2_bias_copy(2,0);
                        // R_lidar2cam2_bias_temp(2, 1)=T_lidar2cam2_bias_copy(2,1);
                        // R_lidar2cam2_bias_temp(2, 2)=T_lidar2cam2_bias_copy(2,2);
                        R_lidar2cam2_bias_temp = rot_dx.matrix() * rot_dy.matrix() * rot_dz.matrix() * T_lidar2cam2_bias.block<3, 3>(0, 0);
                        T_lidar2cam2_bias_copy(0, 0) = R_lidar2cam2_bias_temp(0, 0);
                        T_lidar2cam2_bias_copy(0, 1) = R_lidar2cam2_bias_temp(0, 1);
                        T_lidar2cam2_bias_copy(0, 2) = R_lidar2cam2_bias_temp(0, 2);
                        T_lidar2cam2_bias_copy(1, 0) = R_lidar2cam2_bias_temp(1, 0);
                        T_lidar2cam2_bias_copy(1, 1) = R_lidar2cam2_bias_temp(1, 1);
                        T_lidar2cam2_bias_copy(1, 2) = R_lidar2cam2_bias_temp(1, 2);
                        T_lidar2cam2_bias_copy(2, 0) = R_lidar2cam2_bias_temp(2, 0);
                        T_lidar2cam2_bias_copy(2, 1) = R_lidar2cam2_bias_temp(2, 1);
                        T_lidar2cam2_bias_copy(2, 2) = R_lidar2cam2_bias_temp(2, 2);

                        //Both adjusting tran and angle
                        T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3) + (- iterate_tra_step_small[iii] + dx * iterate_tra_step_small[iii]) * config["tx_plus"].as<float>();//0.002
                        T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3) + (- iterate_tra_step_small[iii] + dy * iterate_tra_step_small[iii]) * config["ty_plus"].as<float>();
                        T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3) + (- iterate_tra_step_small[iii] + dz * iterate_tra_step_small[iii]) * config["tz_plus"].as<float>();

                        //Only update angle
                        // T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3); // - iterate_tra_step_small + dx * iterate_tra_step_small;//0.002
                        // T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3); // - iterate_tra_step_small + dy * iterate_tra_step_small;
                        // T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3); // - iterate_tra_step_small + dz * iterate_tra_step_small;

                        float score = 0;
                        // float score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        // if(ppp<pcs.size()-2 && config["3frame_score"].as<bool>())
                        // {
                        //     score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image)
                        //             +countScore(pc_features_local[ppp+1], distance_images_local[ppp+1], T_lidar2cam2_bias_copy, T_cam2image)
                        //             +countScore(pc_features_local[ppp+2], distance_images_local[ppp+2], T_lidar2cam2_bias_copy, T_cam2image))/3.0;
                        // }
                        // else
                        // {
                        //     if(got_fine_result)
                        //     {
                        //         score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        //     }
                        //     else
                        //     {
                        //         score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        //     }
                        //     // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        // }
                        if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
                        {
                            if (got_fine_result)
                            {
                                score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                            }
                            else
                            {
                                score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                            }
                        }
                        else
                        {
                            // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                            if (got_fine_result)
                            {
                                score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                            }
                            else
                            {
                                score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                            }
                        }
                        omp_set_lock(&mylock);
                        {
                            if (score > max_score)
                            {
                                flag = true;
                                // i = 0;
                                max_score = score;
                                T_lidar2cam2_bias_small(0, 0) = T_lidar2cam2_bias_copy(0, 0);
                                T_lidar2cam2_bias_small(0, 1) = T_lidar2cam2_bias_copy(0, 1);
                                T_lidar2cam2_bias_small(0, 2) = T_lidar2cam2_bias_copy(0, 2);
                                T_lidar2cam2_bias_small(0, 3) = T_lidar2cam2_bias_copy(0, 3);
                                T_lidar2cam2_bias_small(1, 0) = T_lidar2cam2_bias_copy(1, 0);
                                T_lidar2cam2_bias_small(1, 1) = T_lidar2cam2_bias_copy(1, 1);
                                T_lidar2cam2_bias_small(1, 2) = T_lidar2cam2_bias_copy(1, 2);
                                T_lidar2cam2_bias_small(1, 3) = T_lidar2cam2_bias_copy(1, 3);
                                T_lidar2cam2_bias_small(2, 0) = T_lidar2cam2_bias_copy(2, 0);
                                T_lidar2cam2_bias_small(2, 1) = T_lidar2cam2_bias_copy(2, 1);
                                T_lidar2cam2_bias_small(2, 2) = T_lidar2cam2_bias_copy(2, 2);
                                T_lidar2cam2_bias_small(2, 3) = T_lidar2cam2_bias_copy(2, 3);
                                T_lidar2cam2_bias_small(3, 0) = T_lidar2cam2_bias_copy(3, 0);
                                T_lidar2cam2_bias_small(3, 1) = T_lidar2cam2_bias_copy(3, 1);
                                T_lidar2cam2_bias_small(3, 2) = T_lidar2cam2_bias_copy(3, 2);
                                T_lidar2cam2_bias_small(3, 3) = T_lidar2cam2_bias_copy(3, 3);
                                better_cnt++;
                            }
                        }
                        omp_unset_lock(&mylock);
                    }
                    T_lidar2cam2_bias = T_lidar2cam2_bias_small;
                    std::cout << "better_cnt second = " << better_cnt << std::endl;

                    T_lidar2cam_top3 = T_lidar2cam2_bias.topRows(3);
                    T_lidar2image = T_cam2image * T_lidar2cam_top3;
                } //while loop end
                
                if(iii==0){
                    float white_score;
                    got_fine_result = isWhiteEnough(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, white_score);
                    if(!got_fine_result){
                        cout << "1.Optimization failed!!!!!!!!!" << white_score << endl;
                        iii = 2;
                    } 
                    else cout << "1.Success!" << white_score << endl;
                }
                else{
                    float white_score;
                    got_fine_result = isWhiteEnough(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, white_score);
                    if(!got_fine_result) cout << "2.Optimization failed!!!!!!!!!" << white_score << endl;
                    else cout << "2.Success!" << white_score << endl;
                }
            } // 搜索法优化结束

            if(config["visual_search"].as<bool>()){
                pcl::PointCloud<pcl::PointXYZI>::Ptr search_result(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::transformPointCloud(*pcs[ppp], *search_result, T_lidar2cam2_unbias.inverse() * T_lidar2cam2_bias); 
                pcl::visualization::PCLVisualizer viewer("Search Viewer");
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> truth(Real_truth_cloud[ppp], 0, 255, 0);  // GREEN
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> before(pcs[ppp], 255, 0, 0); // BLUE
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> after(search_result, 0, 0, 255); // BLUE
                
                viewer.setBackgroundColor(0,0,0);
                viewer.addPointCloud(Real_truth_cloud[ppp], truth, "truth_cloud");
                viewer.addPointCloud(pcs[ppp], before, "before_cloud");
                viewer.addPointCloud(search_result, after, "after_cloud");

                while (!viewer.wasStopped()){
                    viewer.spinOnce(100);
                    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
                }
            }

            T_lidar2cam2_bias_vec.push_back(T_lidar2cam2_bias);
            if (T_lidar2cam2_bias_vec.size() >= 3)
            {
                Eigen::Matrix3f R_lidar2cam2_bias_delt1 = T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 2].block(0, 0, 3, 3).inverse() * T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 1].block(0, 0, 3, 3);
                Eigen::Matrix3f R_lidar2cam2_bias_delt2 = T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 3].block(0, 0, 3, 3).inverse() * T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 1].block(0, 0, 3, 3);
                Eigen::Vector3f R_euler_xyz_delt1 = R_lidar2cam2_bias_delt1.eulerAngles(0, 1, 2); //x y z
                Eigen::Vector3f R_euler_xyz_delt2 = R_lidar2cam2_bias_delt2.eulerAngles(0, 1, 2); //x y z
                for (int r = 0; r < 3; ++r)
                {
                    if ((R_euler_xyz_delt1[r] > 0) && (abs(R_euler_xyz_delt1[r] - 3) < 0.2))
                    {
                        R_euler_xyz_delt1[r] = (R_euler_xyz_delt1[r] - M_PI) * 180.0 / M_PI;
                    }
                    if ((R_euler_xyz_delt1[r] < 0) && (abs(R_euler_xyz_delt1[r] + 3) < 0.2))
                    {
                        R_euler_xyz_delt1[r] = (R_euler_xyz_delt1[r] + M_PI) * 180.0 / M_PI;
                    }
                    if ((R_euler_xyz_delt2[r] > 0) && (abs(R_euler_xyz_delt2[r] - 3) < 0.2))
                    {
                        R_euler_xyz_delt2[r] = (R_euler_xyz_delt2[r] - M_PI) * 180.0 / M_PI;
                    }
                    if ((R_euler_xyz_delt2[r] < 0) && (abs(R_euler_xyz_delt2[r] + 3) < 0.2))
                    {
                        R_euler_xyz_delt2[r] = (R_euler_xyz_delt2[r] + M_PI) * 180.0 / M_PI;
                    }
                }
                std::cout << "R euler xyz delt 1 " << R_euler_xyz_delt1 * 180.0 / M_PI << std::endl;
                std::cout << "R euler xyz delt 2 " << R_euler_xyz_delt2 * 180.0 / M_PI << std::endl;
                std::cout << "R euler delt = " << abs(R_euler_xyz_delt1[0] - R_euler_xyz_delt2[0]) * 180.0 / M_PI << " " << abs(R_euler_xyz_delt1[1] - R_euler_xyz_delt2[1]) * 180.0 / M_PI << " " << abs(R_euler_xyz_delt1[2] - R_euler_xyz_delt2[2]) * 180.0 / M_PI << std::endl;
                // if((abs(R_euler_xyz_delt1[0]-R_euler_xyz_delt2[0]) * 180.0 / M_PI  < 1) && (abs(R_euler_xyz_delt1[1]-R_euler_xyz_delt2[1]) * 180.0 / M_PI  < 1) &&
                // (abs(R_euler_xyz_delt1[2]-R_euler_xyz_delt2[2]) * 180.0 / M_PI < 1) &&
                // isWhiteEnough(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image))


                // bool big_step = false;
                // float score;
                // if (got_fine_result)
                // {
                //     big_step = isWhiteEnough(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, score);
                // }
                // else
                // {
                //     big_step = isWhiteEnough(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, score);
                // }

                // float confidence_step = countConfidence(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);

                // if (big_step)
                // // if(confidence_step>0.7)
                // {
                //     got_fine_result = true;
                //     large_small_step.push_back(1);
                //     iterate_ang_step_big = 0.005;
                //     iterate_tra_step_big = 0.002; //0.001;
                //     iterate_ang_step_small = 0.002;
                //     iterate_tra_step_small = 0.001; //0.001;
                //     std::cout << "!!!!!!!!Enter!!!!!!! " << R_euler_xyz_delt1[0] - R_euler_xyz_delt2[0] << std::endl;
                // }
                // else //if(ppp<=21)
                // {
                //     got_fine_result = false;
                //     large_small_step.push_back(0);
                //     iterate_ang_step_big = 0.01;
                //     iterate_tra_step_big = 0.002; //0.002;
                //     iterate_ang_step_small = 0.005;
                //     iterate_tra_step_small = 0.001; //0.001;
                // }
            }
            project2image(pc_features_local[ppp], images_withouthist[ppp], image_before_optimize, T_lidar2cam2_bias, T_cam2image);



            //g2o优化
            typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
            typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
            auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            g2o::SparseOptimizer optimizer; 
            optimizer.setAlgorithm(solver); 
            optimizer.setVerbose(true);     

            VertexPose *v = new VertexPose();
            v->setId(0);
            v->setEstimate(AutoCalib::toSE3d(T_lidar2cam2_bias));
            optimizer.addVertex(v);

            for (int i = 0; i < pc_features_local[ppp]->size(); i++)
            {
                EdgeProjectionPoseOnly *e = new EdgeProjectionPoseOnly(pc_features_local[ppp]->points[i], distance_images_local_thin[ppp], T_cam2image, config["add_dis_weight"].as<bool>());

                e->setId(i);
                e->setVertex(0, v);
                e->setMeasurement(0);
                e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
                optimizer.addEdge(e);
            }

            cout << "optimizing ..." << endl;
            optimizer.initializeOptimization();
            optimizer.optimize(30);

            cout << "End optimization" << endl;

            T_lidar2cam2_bias = AutoCalib::toMatrix4f(v->estimate().matrix());

            float confidence = countConfidence(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
            confidence_vec.push_back(confidence);

            std::cout << "Better rate = " << better_cnt / (27 * 27) / 2 << std::endl;
            std::cout << "Picture: " << ppp << std::endl;
            std::cout << "Time: " << omp_get_wtime() - starTime << std::endl;
            float aft_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            std::cout << "After Calibration score  =  " << aft_score << std::endl;
            std::cout << std::endl;
            // rectangle(distance_images_local[ppp],Point(boxes[i].x,boxes[i].y),
            //         Point((boxes[i].x+boxes[i].width),(boxes[i].y+boxes[i].height)),
            //         Scalar(blue,green,red),2,8,0);                                //draw boxes

            std::ostringstream str;
            str << aft_score;
            cv::String label = cv::String(str.str());
            int baseLine = 0;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            putText(distance_images_local[ppp], label, cv::Point(0, 0), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 1), 2);
        }

        R_lidar2cam2_bias(0, 0) = T_lidar2cam2_bias(0, 0);
        R_lidar2cam2_bias(0, 1) = T_lidar2cam2_bias(0, 1);
        R_lidar2cam2_bias(0, 2) = T_lidar2cam2_bias(0, 2);
        R_lidar2cam2_bias(1, 0) = T_lidar2cam2_bias(1, 0);
        R_lidar2cam2_bias(1, 1) = T_lidar2cam2_bias(1, 1);
        R_lidar2cam2_bias(1, 2) = T_lidar2cam2_bias(1, 2);
        R_lidar2cam2_bias(2, 0) = T_lidar2cam2_bias(2, 0);
        R_lidar2cam2_bias(2, 1) = T_lidar2cam2_bias(2, 1);
        R_lidar2cam2_bias(2, 2) = T_lidar2cam2_bias(2, 2);

        // Eigen::Matrix4f T_v = T_lidar2cam2_unbias.inverse() * T_lidar2cam2_bias; //������ת��T_v����  4710�� cy
        // T_v_vec.push_back(T_v);
        // pcl::PointCloud<pcl::PointXYZI>::Ptr infer_cloud_every(new pcl::PointCloud<pcl::PointXYZI>);
        // pcl::transformPointCloud(*pc_features_local[ppp + 1], *infer_cloud_every, T_v); //���������ppp+1��������ģ����?������ô��
        // T_v_infer_cloud.push_back(infer_cloud_every);

        Eigen::Matrix3f R_calibrated_delt = R_gt.inverse() * R_lidar2cam2_bias;
        Eigen::Vector3f euler_ang_delt_zyx = R_calibrated_delt.eulerAngles(2, 1, 0); //zyx
        Eigen::Vector3f euler_ang_delt_xyz = R_calibrated_delt.eulerAngles(0, 1, 2); //xyz

        Eigen::Matrix3f R_calibrated_inv_delt = R_lidar2cam2_bias.inverse() * R_gt;
        Eigen::Vector3f euler_ang_inv_delt_zyx = R_calibrated_inv_delt.eulerAngles(2, 1, 0); //zyx
        Eigen::Vector3f euler_ang_inv_delt_xyz = R_calibrated_inv_delt.eulerAngles(0, 1, 2); //xyz

        euler_ang_delt_zyx_vec.push_back(euler_ang_delt_zyx);
        euler_ang_delt_xyz_vec.push_back(euler_ang_delt_xyz);
        euler_ang_inv_delt_zyx_vec.push_back(euler_ang_inv_delt_zyx);
        euler_ang_inv_delt_xyz_vec.push_back(euler_ang_inv_delt_xyz);
        std::cout << "1== " << euler_ang_delt_zyx[0] * 180.0 / M_PI << " " << euler_ang_delt_zyx[1] * 180.0 / M_PI << " " << euler_ang_delt_zyx[2] * 180.0 / M_PI << std::endl;
        std::cout << "2== " << euler_ang_delt_xyz[0] * 180.0 / M_PI << " " << euler_ang_delt_xyz[1] * 180.0 / M_PI << " " << euler_ang_delt_xyz[2] * 180.0 / M_PI << std::endl;
        std::cout << "3== " << euler_ang_inv_delt_zyx[0] * 180.0 / M_PI << " " << euler_ang_inv_delt_zyx[1] * 180.0 / M_PI << " " << euler_ang_inv_delt_zyx[2] * 180.0 / M_PI << std::endl;
        std::cout << "4== " << euler_ang_inv_delt_xyz[0] * 180.0 / M_PI << " " << euler_ang_inv_delt_xyz[1] * 180.0 / M_PI << " " << euler_ang_inv_delt_xyz[2] * 180.0 / M_PI << std::endl;

        Eigen::Vector3f euler_ang_calibrated_zyx = R_lidar2cam2_bias.eulerAngles(0, 1, 2); //x y z
        euler_ang_calibrated_zyx_vec.push_back(euler_ang_calibrated_zyx);

        std::cout << "euler angle calibrated = " << std::endl
                  << euler_ang_calibrated_zyx * 180.0 / M_PI << std::endl
                  << "euler angle gt = " << std::endl
                  << euler_ang_gt_zyx * 180.0 / M_PI << std::endl;

        std::cout << "euler angle delt = " << std::endl
                  << euler_ang_delt_xyz[0] * 180.0 / M_PI << " " << euler_ang_delt_xyz[1] * 180.0 / M_PI
                  << " " << euler_ang_delt_xyz[2] * 180.0 / M_PI << std::endl;
        std::cout << std::endl;

        T_lidar2cam2_bias_last = T_lidar2cam2_bias; //lidar2cam last

        

        Eigen::Matrix4f T_v = T_lidar2cam2_unbias.inverse() * T_lidar2cam2_bias_last;
        T_v_vec.push_back(T_v);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr infer_cloud_every(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::transformPointCloud(*pcs[ppp], *infer_cloud_every, T_v);   
        T_v_infer_cloud.push_back(infer_cloud_every);
        cout << "infer_cloud in success!"<<endl;

        cv::Mat image2;
        if (got_fine_result)
        {
            project2image(pc_features_local[ppp], distance_images_local_thin[ppp], image2, T_lidar2cam2_bias, T_cam2image);
        }
        else
        {
            project2image(pc_features_local[ppp], distance_images_local[ppp], image2, T_lidar2cam2_bias, T_cam2image);
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr line_features(new pcl::PointCloud<pcl::PointXYZI>);

        // ��ȡ�ı�����
        std::ifstream inFile("../data" + std::to_string(frame_cnt) + "/pc_lines");
        std::string lineStr; // �ļ��е�һ������

        if (inFile) // �и��ļ�
        {
            int i = 0;                       // ѭ���±�
            while (getline(inFile, lineStr)) // line�в�����ÿ�еĻ��з�
            {
                // stringתchar *
                char *lineCharArray;
                const int len = lineStr.length();
                lineCharArray = new char[len + 1];
                strcpy(lineCharArray, lineStr.c_str());

                char *p;                        // �ָ�����ַ���??
                p = strtok(lineCharArray, " "); // ����spaceChar�ָ�
                std::vector<double> data_temp;
                // �����ݼ���vector��
                while (p)
                {
                    data_temp.push_back(atof(p));
                    // int a = strlen(p);
                    p = strtok(NULL, " ");
                }
                pcl::PointXYZI pc_temp;
                pc_temp.x = data_temp[0];
                pc_temp.y = data_temp[1];
                pc_temp.z = data_temp[2];
                line_features->push_back(pc_temp);
                data_temp.clear();
            }
        }

        cv::Mat image_line_feature;
        project2image(line_features, gray_image_vec[0], image_line_feature, T_lidar2cam2_unbias, T_cam2image);
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        cv::Mat image4, image_result;
        filterUnusedPoiintCloudFeature(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image, pc_feature_filtered);
        project2image(pc_feature_filtered, images[ppp], image4, T_lidar2cam2_bias, T_cam2image);
        project2image(filtered_pc[ppp], images[ppp], image_result, T_lidar2cam2_unbias, T_cam2image);

        cv::Mat after_on_image, after_on_semantic;
        project2image(pc_features_local[ppp], images[ppp], after_on_image, T_lidar2cam2_bias, T_cam2image);
        project2image(pc_features_local[ppp], distance_images_local_thin[ppp], after_on_semantic, T_lidar2cam2_bias, T_cam2image);

        cv::Mat thin_pcf;
        project2image(pc_features[ppp], distance_images_local_thin[ppp], thin_pcf, T_lidar2cam2_bias, T_cam2image);

        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/project_result_" + std::to_string(ppp) + ".png", image_result);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/original_bias_" + std::to_string(ppp) + ".png", image2);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/unbias_" + std::to_string(ppp) + ".png", image_unbias);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/calibrated_result_" + std::to_string(ppp) + ".png", image4);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/fat_" + std::to_string(ppp) + ".png", distance_images_local[ppp]);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/thin_" + std::to_string(ppp) + ".png", distance_images_local_thin[ppp]);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/thin_pcf_" + std::to_string(ppp) + ".png", thin_pcf);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/full_pcf_" + std::to_string(ppp) + ".png", fullimg);

        cv::Mat dst;
        dst.create(image1.rows * 5, image1.cols, CV_8UC3);

        before_on_semantic.copyTo(dst(cv::Rect(0, 0, images[ppp].cols, images[ppp].rows)));
        after_on_semantic.copyTo(dst(cv::Rect(0, image1.rows * 1, images[ppp].cols, images[ppp].rows)));
        before_on_image.copyTo(dst(cv::Rect(0, image1.rows * 2, images[ppp].cols, images[ppp].rows)));
        after_on_image.copyTo(dst(cv::Rect(0, image1.rows * 3, images[ppp].cols, images[ppp].rows)));
        image_unbias.copyTo(dst(cv::Rect(0, image1.rows * 4, images[ppp].cols, images[ppp].rows)));
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/display" + std::to_string(ppp) + ".png", dst);
        
  

        // cv::namedWindow(result_file[ppp], cv::WINDOW_NORMAL);
        // cv::imshow(result_file[ppp], dst);
        // std::ostringstream str;
        //   str << min_dist;
        //   marker.text = str.str();
        // cv::imwrite(result_file[ppp], dst);
        // cv::waitKey(0);
        calibrated_result_vec.push_back(T_lidar2cam2_bias);  //ppp�����һ��ʹ��?

    }
    
    //mark ��ppp

    if (config["save_calibrated_result"].as<bool>())
    {
        if (calibrated_result)
        {
            calibrated_result << endl << endl;
            calibrated_result << "Message_filter_cloud.size()= "<<Message_filter_cloud.size();
            calibrated_result << "\tT_v_infer_cloud.size()= "<<T_v_infer_cloud.size();
            calibrated_result << "\tReal_truth_cloud.size()= "<<Real_truth_cloud.size() << endl << endl;


            #ifdef Dists_evaluation
            std::vector<float> score1;
            std::vector<float> score2;
            float avg_1 = 0, avg_2 = 0, avg_2_full = 0;
            #endif

            #ifdef NDT_evaluation
            std::vector<float> score3;
            std::vector<float> score4;
            float avg_3 = 0, avg_4 = 0, avg_4_full = 0;
            #endif

            #ifdef ICP_evaluation
            std::vector<float> score5;
            std::vector<float> score6;
            float avg_5 = 0, avg_6 = 0, avg_6_full = 0;
            #endif

            for(int i=0;i< T_v_infer_cloud.size();++i) 
            {
                if( (i<Message_filter_cloud.size()  )&& (i<Real_truth_cloud.size()) )
                {
                    evaluation_function(T_v_infer_cloud[i],  Message_filter_cloud[i], Real_truth_cloud[i],score_count_cloud);
                    #ifdef Dists_evaluation
                    score1.push_back(score_count_cloud[0]);
                    score2.push_back(score_count_cloud[1]);
                    #endif
                    #ifdef NDT_evaluation
                    score3.push_back(score_count_cloud[2]);
                    score4.push_back(score_count_cloud[3]);
                    #endif
                    #ifdef ICP_evaluation
                    score5.push_back(score_count_cloud[4]);
                    score6.push_back(score_count_cloud[5]);
                    #endif
                }
            }

            #ifdef Dists_evaluation
            calibrated_result << "Dists_evaluation has been activated!" << endl;
            calibrated_result << "i\t" << "infer_cloud_score\t" << "filter_cloud_score" << endl;
            for(int i=0 ; i < score1.size() ; ++i) 
            {
                avg_1 += score1[i];
                if(i%2 == 0) avg_2 += score2[i];
                avg_2_full += score2[i];
                calibrated_result << i << "\t" << score1[i] << "\t\t" << score2[i] << endl;
            }
            avg_1 /= score1.size();
            avg_2_full /= score2.size();
            avg_2 /= (score2.size()/2);

            calibrated_result << "avg_1 = " << avg_1 << "    avg_2 = " << avg_2 << "    avg_2_full = " << avg_2_full << endl << endl;
            #endif

            #ifdef NDT_evaluation
            calibrated_result << "NDT_evaluation has been activated!" << endl;
            calibrated_result << "i\t" << "infer_cloud_score\t" << "filter_cloud_score" << endl;
            for(int i=0 ; i < score3.size() ; ++i) 
            {          
                avg_3 += score3[i];
                if(i%2 == 0) avg_4 += score4[i];
                avg_4_full += score4[i];
                calibrated_result << i << "\t" << score3[i] << "\t\t\t" << score4[i] << endl;
            }
            avg_3 /= score3.size();
            avg_4_full /= score4.size();
            avg_4 /= (score4.size()/2);

            calibrated_result << "avg_3 = " << avg_3 << "    avg_4 = " << avg_4 << "    avg_4_full = " << avg_4_full << endl << endl;
            #endif

            #ifdef ICP_evaluation
            calibrated_result << "ICP_evaluation has been activated!" << endl;
            calibrated_result << "i\t" << "infer_cloud_score\t" << "filter_cloud_score" << endl;
            for(int i=0 ; i < score5.size() ; ++i) 
            {          
                avg_5 += score5[i];
                if(i%2 == 0) avg_6 += score6[i];
                avg_6_full += score6[i];
                calibrated_result << i << "\t" << score5[i] << "\t\t" << score6[i] << endl;
            }
            avg_5 /= score5.size();
            avg_6_full /= score6.size();
            avg_6 /= (score6.size()/2);

            calibrated_result << "avg_5 = " << avg_5 << "    avg_6 = " << avg_6 << "    avg_6_full = " << avg_6_full << endl << endl;
            #endif

            // calibrated_result << "i\t"  << "score1\t\t" << "score2\t\t" << "score3\t\t" << "score4\t\t" << "score5\t\t" << "score6\t\t" << endl;

            // for(int i=0;i< T_v_infer_cloud.size();++i) 
            // {
            //         if( (i<Message_filter_cloud.size()  )&& (i<truth_cloud_copy_from_pcs.size()) )

            //             evaluation_function(T_v_infer_cloud[i],  Message_filter_cloud[i], truth_cloud_copy_from_pcs[i],score_count_cloud);

                        
            //                 calibrated_result << i << "\t";
            //                 calibrated_result << score_count_cloud[0] << "\t\t"; 
            //                 calibrated_result << score_count_cloud[1] << "\t\t";
            //                 calibrated_result << score_count_cloud[2] << "\t\t";
            //                 calibrated_result << score_count_cloud[3] << "\t\t";
            //                 calibrated_result << score_count_cloud[4] << "\t\t";
            //                 calibrated_result << score_count_cloud[5] << endl;
            // }

            calibrated_result << "x_c=[";
            for (int i = 0; i < euler_ang_calibrated_zyx_vec.size(); ++i)
            {
                calibrated_result << euler_ang_calibrated_zyx_vec[i][2] * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;
            calibrated_result << "y_c=[";
            for (int i = 0; i < euler_ang_calibrated_zyx_vec.size(); ++i)
            {
                calibrated_result << euler_ang_calibrated_zyx_vec[i][1] * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;
            calibrated_result << "z_c=[";
            for (int i = 0; i < euler_ang_calibrated_zyx_vec.size(); ++i)
            {
                calibrated_result << euler_ang_calibrated_zyx_vec[i][0] * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;
        }
        else
        {
            std::cout << "save error" << std::endl;
        }

        if (calibrated_result)
        {
            calibrated_result << "x2_c=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                if (abs(abs(euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI > 0)
                {
                    euler_ang_delt_zyx_vec[i][2] = euler_ang_delt_zyx_vec[i][2] - M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI << ",";
                }
                else if (abs(abs(euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI < 0)
                {
                    euler_ang_delt_zyx_vec[i][2] = euler_ang_delt_zyx_vec[i][2] + M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI << ",";
                }
                else
                {
                    calibrated_result << euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "y2_c=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                if (abs(abs(euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI > 0)
                {
                    euler_ang_delt_zyx_vec[i][1] = euler_ang_delt_zyx_vec[i][1] - M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI << ",";
                }
                else if (abs(abs(euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI < 0)
                {
                    euler_ang_delt_zyx_vec[i][1] = euler_ang_delt_zyx_vec[i][1] + M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI << ",";
                }
                else
                {
                    calibrated_result << euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "z2_c=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                if (abs(abs(euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI > 0)
                {
                    euler_ang_delt_zyx_vec[i][0] = euler_ang_delt_zyx_vec[i][0] - M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI << ",";
                }
                else if (abs(abs(euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI < 0)
                {
                    euler_ang_delt_zyx_vec[i][0] = euler_ang_delt_zyx_vec[i][0] + M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI << ",";
                }
                else
                {
                    calibrated_result << euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl
                              << std::endl;

            float mean_x = 0, mean_y = 0, mean_z = 0;
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                mean_x = mean_x + euler_ang_delt_zyx_vec[i][2];
                mean_y = mean_y + euler_ang_delt_zyx_vec[i][1];
                mean_z = mean_z + euler_ang_delt_zyx_vec[i][0];
                std::cout << "mean x = " << mean_x << std::endl;
            }
            mean_x = mean_x / euler_ang_delt_zyx_vec.size();
            mean_y = mean_y / euler_ang_delt_zyx_vec.size();
            mean_z = mean_z / euler_ang_delt_zyx_vec.size();
            std::cout << "mean x = " << mean_x << std::endl;

            calibrated_result << "roll=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                calibrated_result << (euler_ang_delt_zyx_vec[i][2] - mean_x) * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "pitch=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                calibrated_result << (euler_ang_delt_zyx_vec[i][1] - mean_y) * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "yaw=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                calibrated_result << (euler_ang_delt_zyx_vec[i][0] - mean_z) * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl
                              << std::endl;

            calibrated_result << "mean = " << mean_x << " " << mean_y << " " << mean_z << std::endl;
            float mean_x_new = 0, mean_y_new = 0, mean_z_new = 0;
            for (int i = euler_ang_delt_zyx_vec.size() / 4; i < euler_ang_delt_zyx_vec.size() / 4 * 3; ++i)
            {
                mean_x_new += (euler_ang_delt_zyx_vec[i][2] - mean_x);
                mean_y_new += (euler_ang_delt_zyx_vec[i][1] - mean_y);
                mean_z_new += (euler_ang_delt_zyx_vec[i][0] - mean_z);
            }
            mean_x_new = mean_x_new / euler_ang_delt_zyx_vec.size();
            mean_y_new = mean_y_new / euler_ang_delt_zyx_vec.size();
            mean_z_new = mean_z_new / euler_ang_delt_zyx_vec.size();
            calibrated_result << "mean new= " << mean_x_new << " " << mean_y_new << " " << mean_z_new << std::endl;

            calibrated_result << std::endl;
            // calibrated_result << "large_small_step=[0,0,";
            // for (int i = 0; i < large_small_step.size(); ++i)
            // {
            //     calibrated_result << large_small_step[i] << ",";
            // }
            // calibrated_result << "]" << std::endl
            //                   << std::endl;

            float confidence_mean = 0;
            calibrated_result << "confidence=[";
            for (int i = 0; i < confidence_vec.size(); ++i)
            {
                calibrated_result << confidence_vec[i] << ",";
                confidence_mean += confidence_vec[i];
            }
            calibrated_result << "]" << std::endl
                              << std::endl;
            confidence_mean /= confidence_vec.size();
            // calibrated_result << "roll_cfd_mean=[";
            // for (int i = 0; i < euler_ang_delt_zyx_vec.size() - 1; ++i)
            // {
            //     if (large_small_step[i] == 1 && large_small_step[i + 1] == 1)
            //     {
            //         calibrated_result << (euler_ang_delt_zyx_vec[i][2] - mean_x) * 180.0 / M_PI << ",";
            //     }
            // }
            // calibrated_result << "]" << std::endl;

            // calibrated_result << "pitch_cfd_mean=[";
            // for (int i = 0; i < euler_ang_delt_zyx_vec.size() - 1; ++i)
            // {
            //     if (large_small_step[i] == 1 && large_small_step[i + 1] == 1)
            //     {
            //         calibrated_result << (euler_ang_delt_zyx_vec[i][1] - mean_y) * 180.0 / M_PI << ",";
            //     }
            // }
            // calibrated_result << "]" << std::endl;

            // calibrated_result << "yaw_cfd_mean=[";
            // for (int i = 0; i < euler_ang_delt_zyx_vec.size() - 1; ++i)
            // {
            //     if (large_small_step[i] == 1 && large_small_step[i + 1] == 1)
            //     {
            //         calibrated_result << (euler_ang_delt_zyx_vec[i][0] - mean_z) * 180.0 / M_PI << ",";
            //     }
            // }
            // calibrated_result << "]" << std::endl
            //                   << std::endl;
        }
        else
        {
            std::cout << "save error" << std::endl;
        }

        if (
            calibrated_result)
        {   
            for (int i = 0; i < T_v_vec.size(); ++i)
            {
                calibrated_result <<"T_v "<< i<<": "<< T_v_vec[i] << std::endl;//�洢������?������
                
            }

            for(int i = 0; i < calibrated_result_vec.size(); ++i) {
                calibrated_result <<"calibrated_result_T  "<< i<<": "<< calibrated_result_vec[i] << std::endl;
            }
            
            
        }
    }
    calibrated_result.close();

    /****************************************************************/

    //    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //
    //    viewer->initCameraParameters ();
    //    int v1(0);
    //
    //    viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    //
    //    viewer->setBackgroundColor (0, 0, 0, v1);
    //
    //    viewer->addText ("Radius: 0.01", 10, 10, "v1 text", v1);
    //
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb(raw);
    //
    //    viewer->addPointCloud<pcl::PointXYZI> (cloud, rgb, "sample cloud1", v1);
    //    int v2(0);
    //
    //    viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    //
    //    viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
    //
    //    viewer->addText ("Radius: 0.1", 10, 10, "v2 text", v2);
    //
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color (cloud, 0, 255, 0);
    //
    //    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud2", v2);

    //    pcl::visualization::PCLVisualizer viewer;
    //    int a, b, c, d;
    //    viewer.createViewPort (0.0, 0.0, 0.5, 0.5, a); //(Xmin,Ymin,Xmax,Ymax)���ô�������
    //    viewer.createViewPort (0.5, 0.0, 1.0, 0.5, b);
    //    viewer.createViewPort (0, 0.5, 0.5, 1.0, c);
    //    viewer.createViewPort (0.5, 0.5, 1.0, 1.0, d);
    //    viewer.addPointCloud<pcl::PointXYZI>(raw, "cloud1", a);
    //    viewer.addPointCloud<pcl::PointXYZI>(filtered4, "cloud2", b);
    //    viewer.addPointCloud<pcl::PointXYZI>(filtered_noground, "cloud3",c);
    //    viewer.addPointCloud<pcl::PointXYZI>(edges, "cloud4",d);
    //    std::cout << "Showing" << std::endl;
    //
    //    viewer.spin();
    //    pcl::visualization::CloudViewer viewer1("vision");
    //    pcl::visualization::CloudViewer viewer2("vision2");
    //    viewer.showCloud(raw);
    //    viewer2.showCloud(filtered2);
}

void AutoCalib::PerformNdt(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &in_parent_cloud_vec,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &in_child_cloud_vec,
                           int calib_frame_cnt, Eigen::Matrix4f &pcs_current_guess,
                           std::vector<Eigen::Matrix4f> &pcs_calib)
{

    int cnt = std::min(in_parent_cloud_vec.size(), in_child_cloud_vec.size());
    std::cout << "transformation from lidar " << calib_frame_cnt + 1 << " to lidar 1" << std::endl;

    for (int i = 0; i < cnt; i++)
    {
        pcl::console::TicToc time; //����ʱ����?
        time.tic();                //time.tic��ʼ  time.toc����ʱ��

        std::cout << "PointCloud Frame" << i << endl;
        //Initializing Normal Distributions Transform (NDT).
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize(0.05, 0.05, 0.05);
        approximate_voxel_filter.setInputCloud(in_child_cloud_vec[i]);
        approximate_voxel_filter.filter(*filtered_cloud);

        pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

        ndt.setTransformationEpsilon(0.1);
        ndt.setStepSize(0.1);
        ndt.setResolution(0.5);

        ndt.setMaximumIterations(35);

        ndt.setInputSource(filtered_cloud);
        ndt.setInputTarget(in_parent_cloud_vec[i]);

        pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        ndt.align(*output_cloud, pcs_current_guess);
        // pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        // // Set the input source and target
        // icp.setInputCloud(filtered_cloud);
        // icp.setInputTarget(in_parent_cloud_vec[i]);

        // icp.setMaximumIterations(35);
        // pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        // icp.align(*output_cloud);

        std::cout << "Normal Distributions Transform converged:" << ndt.hasConverged()
                  << " score: " << ndt.getFitnessScore() << " Probability " << ndt.getTransformationProbability() << std::endl;

        // Transforming unfiltered, input cloud using found transform.
        //pcl::transformPointCloud(*in_child_cloud_, *output_cloud, ndt.getFinalTransformation());

        Eigen::Matrix4f T_pcs_current = ndt.getFinalTransformation();

        //Eigen::Matrix3f rotation_matrix = T_pcs_current.block(0, 0, 3, 3);
        //Eigen::Vector3f translation_vector = T_pcs_current.block(0, 3, 3, 1);

        //std::cout << "This transformation can be replicated using:" << std::endl;
        //std::cout << "rosrun tf static_transform_publisher " << translation_vector.transpose()
        //    << " " << rotation_matrix.eulerAngles(2, 1, 0).transpose() << " /" << parent_frame_
        //    << " /" << child_frame_ << " 10" << std::endl;

        std::cout << "Corresponding transformation matrix:" << std::endl
                  << std::endl
                  << T_pcs_current << std::endl
                  << std::endl;

        pcs_calib.push_back(T_pcs_current);
        std::cout << time.toc() << " ms" << std::endl;
    }
}
// void AutoCalib::PerformICP(std::vector<cv::Mat> & in_parent_images_vec, std::vector<cv::Mat>& in_child_images_vec,
//                            int calib_frame_cnt,Eigen::Matrix4f & images_current_guess,std::vector<Eigen::Matrix4f > & images_calib)
// {
//     int cnt = std::min(in_parent_images_vec.size(), in_child_images_vec.size());
//     std::cout << "transformation from camera " << calib_frame_cnt + 1 << " to camera 1" << std::endl;

//     for (int i = 0; i < cnt; i++)
//     {
//         //cv::Matx44d  pose;

//         double re,pose[16];
//         cv::ppf_match_3d::ICP calculate(100, 0.005f, 2.5f, 8);
//         calculate.registerModelToScene(in_parent_images_vec[i], in_child_images_vec[i],re,pose);

//         images_calib.push_back();
//     }
// }
//��ǩ��ȫ���Ż�����
void AutoCalib::GlobalOptimize(std::vector<std::vector<Eigen::Matrix4f>> &pcs_calib_vec, std::vector<std::vector<Eigen::Matrix4f>> &Calibrated_Result_vec,
                               std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> &pcs_vec,
                               std::vector<std::vector<cv::Mat>> &images_vec,
                               std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> &pcs_feature_vec,
                               std::vector<std::vector<cv::Mat>> &images_feature_vec, bool &add_dis_weight,
                               std::vector<bool> &overlap_vec,
                               int &calib_frame_num, std::vector<Eigen::Matrix3f> &k_vec)
{
    int cnt = Calibrated_Result_vec[calib_frame_num - 1].size();
    for (int i = 0; i < pcs_calib_vec.size(); i++)
    {
        int a = pcs_calib_vec[i].size();
        int b = Calibrated_Result_vec[i].size();
        cnt = std::min(cnt, a);
        cnt = std::min(cnt, b);
    }
    for (int i = 0; i < cnt; i++)
    {
        // �趨g2o
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer; // ͼģ��
        optimizer.setAlgorithm(solver); // ���������??
        optimizer.setVerbose(true);     // �򿪵������??
        for (int j = 0; j < calib_frame_num; j++)
        {
            VertexPose *v = new VertexPose();
            v->setId(j * 2);
            v->setEstimate(AutoCalib::toSE3d(Calibrated_Result_vec[j][i]));
            optimizer.addVertex(v);
        }
        for (int j = 0; j < calib_frame_num - 1; j++)
        {
            VertexPose *v = new VertexPose();
            v->setId(j * 2 + 1);
            v->setEstimate(AutoCalib::toSE3d(pcs_calib_vec[j + 1][i]));
            optimizer.addVertex(v);
        }
        for (int j = 0; j < calib_frame_num - 1; j++)
        {
            std::vector<cv::KeyPoint> keypoints_first, keypoints_second;
            std::vector<cv::DMatch> matches;
            if (overlap_vec[j + 1] = true)
            {

                AutoCalib::find_feature_matches(images_vec[0][i], images_vec[j + 1][i], keypoints_first, keypoints_second, matches);
            }

            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
            approximate_voxel_filter.setLeafSize(0.05, 0.05, 0.05);
            approximate_voxel_filter.setInputCloud(pcs_vec[j + 1][i]);
            approximate_voxel_filter.filter(*filtered_cloud);

            EdgePRScale *e = new EdgePRScale(pcs_feature_vec[0][i], pcs_feature_vec[j + 1][i],
                                             images_feature_vec[0][i], images_feature_vec[j + 1][i],
                                             pcs_vec[0][i], pcs_vec[j + 1][i],
                                             images_vec[0][i], images_vec[j + 1][i],
                                             k_vec[0], k_vec[j + 1], add_dis_weight, overlap_vec[j + 1], calib_frame_num,
                                             keypoints_first, keypoints_second, matches, filtered_cloud);

            e->setId(j);
            e->setVertex(0, optimizer.vertices()[0]);
            e->setVertex(1, optimizer.vertices()[j * 2 + 1]);
            e->setVertex(2, optimizer.vertices()[(j + 1) * 2]);
            e->setMeasurement(0);
            e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            optimizer.addEdge(e);
        }

        cout << "optimizing ..." << endl;
        optimizer.initializeOptimization();
        optimizer.optimize(20);
        cout << "end optimizing" << endl;

        cout << "saving optimization results ..." << endl;
        for (int j = 0; j < calib_frame_num; j++)
        {

            Eigen::Matrix4f T_after_global_optimze = AutoCalib::toMatrix4f(dynamic_cast<VertexPose *>(optimizer.vertex(j * 2))->estimate().matrix());
            cv::Mat image_after_global_potimze;
            AutoCalib::project2image(pcs_feature_vec[j][i], images_vec[j][i], image_after_global_potimze, T_after_global_optimze, k_vec[j]);
            cv::imwrite("../data" + std::to_string(j) + "/result/after_global_optimze_result_" + std::to_string(i) + ".png", image_after_global_potimze);
            std::string calibrated_result_path = "../data" + std::to_string(j) + "/result/calibrated_result";
            std::ofstream calibrated_result(calibrated_result_path, std::ios::app);
            calibrated_result << "after optimize" << std::endl
                              << "cnt=[" << i << "]" << T_after_global_optimze << std::endl;

            calibrated_result.close();
        }
    }
}
AutoCalib::AutoCalib(const std::string &ConfigFile, int frame_num) //���캯��
{
    config = YAML::LoadFile(ConfigFile); //config�ļ���·��
    debug = config["debug_OnOff"].as<bool>();
    frame_cnt = frame_num;
    overlap = config["overlap"].as<bool>();
    add_dis_weight = config["add_dis_weight"].as<bool>();
    rings = config["rings"].as<int>();
    lowerBound = config["lowerBound"].as<float>();
    upperBound = config["upperBound"].as<float>();
    fx = config["fx"].as<float>();
    fy = config["fy"].as<float>();
    cx = config["cx"].as<float>();
    cy = config["cy"].as<float>();
    //cv::Mat now_img = cv::Mat(cv::Size(512, 480), CV_8UC3);
    //cv::Mat now_img2 = cv::Mat(cv::Size(512, 480), CV_8UC4);
    Eigen::Matrix4f T_lidar2cam0_unbias = Eigen::Matrix4f::Identity();    //lidar2cam
    Eigen::Matrix4f T_lidar2cam0_bias = Eigen::Matrix4f::Identity();      //lidar2cam
    Eigen::Matrix4f T_lidar2cam2_bias = Eigen::Matrix4f::Identity();      //lidar2cam
    Eigen::Matrix4f T_lidar2cam2_bias_last = Eigen::Matrix4f::Identity(); //lidar2cam last
    Eigen::Matrix4f T_lidar2cam2_unbias = Eigen::Matrix4f::Identity();    //lidar2cam
    Eigen::Matrix4f T_cam02cam2 = Eigen::Matrix4f::Identity();            //cam2cam
    T_cam2image << fx, 0.f, cx, 0.f, fy, cy, 0, 0, 1;
    Eigen::Matrix3f rot_icp = Eigen::Matrix3f::Identity();
}
