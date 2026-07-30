folder <b>capstone</b> includes two parts: computer vision and reinforcement learning from the capstone project.

rover_connection_guide.md lists the way to connect and control the rover. 

Future Considerations (7/30/2026) - Connor Benoit
1. Segmentation could still use improvement, I believe. Particularly with respect to shadows, dust, and other environmental hazards.
2. As of now, there are some mislabeling with what each semantic class is, due to the retraining undertaken for mobilenetv3. The code should function fine, its mostly the notes that can cause confusion. I believe TerrainCostMapper is the most accurate labeling mask. 
3. There are still many kinks to iron out with the persistent mapping. I only managed to get it going my last week of the internship. I encourage verification and optimizations from whoever works on it next. The rover may not be using it to its full potential.
4. Science Targeting mode uses "big rocks" and "small rocks." Jake worked on that side of things. I am not sure if small rocks should be an option or how well that meshes with the segmentation model since it only really sees "big rocks" and identifies what would be "small rocks" as rough terrain.
5. Please document and backup everything the best you can.
6. Here are some papers I was provided by Dr. Ramgopal Mettu. They may be of use to anyone getting caught up to speed with regards to autonomous robotics:
   An overview of the lunar rover “system”:
     https://arxiv.org/html/2603.17232v1
   The classic “Active Perception” paper from 1988:
     https://ieeexplore.ieee.org/document/5968
   A revisiting of the topic in 2017:
     https://pmc.ncbi.nlm.nih.gov/articles/PMC6954017/
   And finally, active perception with respect to vision:
     https://arxiv.org/abs/2512.03687v1
