Fast Formula Student random track generator with python binding.

# Build
````
mkdir build
cd build
cmake ..
build
````

# Usage

```` Python
import build.fsgenerator as fsg

imgs, angles = fsg.generate_fsg_tracks(num_tracks=1000000, 
                                       propagation_dist=2.0, 
                                       detection_prob=0.8, 
                                       max_false_positives=20, 
                                       img_range=30, 
                                       img_resolution=0.5)
````