
void AutoCalib::icp_erase(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc1,pcl::PointCloud<pcl::PointXYZI>::Ptr &pc2, Eigen::Matrix4f &T_v,pcl::PointCloud<pcl::PointXYZI>::Ptr &dynamic_cloud) //d????????????????????????????f
{

    float max_range = 0.05;//设置的阈值  2太大，0.002太小,0.01 little,next:(2+0.01)/2=1,1还是大了，动未剔除；（1+0.01）/2=0.5 still big ;  try 0.1 already ;  减小到0.05，只有动目标
    // Transform the input dataset using the final transformation
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*pc2, *output_cloud, T_v.inverse());
    pcl::KdTreeFLANN<pcl::PointXYZI> tree_;
    tree_.setInputCloud(output_cloud); //构建kdtree，来搜索pc1的每个点

    int nearK = 1; //最近临点个数
    
    // For each point in the source dataset
    std::vector<int> del;
    std::vector<int> sum;
    for (size_t i = 0; i < pc1->points.size(); ++i) //遍历pc1的点
    {
        std::vector<int> nn_indices(nearK);
        std::vector<float> nn_dists(nearK);
        // Find its nearest neighbor in the target
        tree_.nearestKSearch(pc1->points[i], nearK, nn_indices, nn_dists);    //查询对象点，距离，邻近点索引，对象点与邻近点的距离平方值
        std::cout<<"nn_indices[0]= "<<nn_indices[0]<<std::endl;
        if (nn_dists[0] >= max_range)
            {
                del.push_back(i);
            }
       else
        { 
             sum.push_back(i);
        }    
    }
    pcl::copyPointCloud(*pc1, del, *dynamic_cloud);  //cloud_in  index  cloud_out ,only choosen index to output 
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*pc1, sum, *tmp_cloud);
    //*pc1=*tmp_cloud;
    pc1=tmp_cloud;
 