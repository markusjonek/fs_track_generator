#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <omp.h>

#include "common.hpp"
#include "track_generator.hpp"

namespace py = pybind11;

typedef std::pair<py::array_t<float>, py::array_t<float>> PyArrayPair;

void fillTrackDataPoint(float* img_ptr, 
                        float& angle, 
                        float propagation_dist, 
                        float detection_prob, 
                        int max_false_positives, 
                        int img_size, 
                        float img_resolution) {

    Track track;
    track.buildRealisticFSTrack(detection_prob, max_false_positives);
    track.renormalizeTrack(propagation_dist);
    
    std::vector<Point> cones;
    track.getAllCones(cones);

    angle = track.getPropagationAngle(propagation_dist);

    for (int i = 0; i < cones.size(); i++) {
        float x_car = cones[i].x / img_resolution;
        float y_car = cones[i].y / img_resolution;
        int x_img = static_cast<int>(img_size / 2.0 - y_car);
        int y_img = static_cast<int>(img_size / 2.0 - x_car);

        if (x_img >= 0 && x_img < img_size && y_img >= 0 && y_img < img_size) {
            img_ptr[y_img * img_size + x_img] = 1.0f;
        }
    }
}


PyArrayPair generateFSDataSet(const int num_tracks,
                              const float propagation_dist, 
                              const float detection_prob, 
                              const int max_false_positives,
                              const float img_range,
                              const float img_resolution) {
 
    const int img_size = static_cast<int>(img_range / img_resolution);

    py::array_t<float> images({num_tracks, img_size, img_size});
    py::array_t<float> angles(num_tracks);

    std::fill(images.mutable_data(), images.mutable_data() + images.size(), 0.0f);

    float* angles_ptr = static_cast<float*>(angles.request().ptr);
    auto img = images.mutable_unchecked<3>();

    #pragma omp parallel for
    for (int i = 0; i < num_tracks; i++) {
        float* img_ptr = &img(i, 0, 0);
        fillTrackDataPoint(img_ptr, angles_ptr[i], propagation_dist, detection_prob, max_false_positives, img_size, img_resolution);
    }
    return std::make_pair(images, angles);
}

PYBIND11_MODULE(fsgenerator, m) {
    m.def("generate_fs_tracks", &generateFSDataSet, "Generates a track dataset",
          py::arg("num_tracks"), py::arg("propagation_dist"), py::arg("detection_prob"), 
          py::arg("max_false_positives"), py::arg("img_range"), py::arg("img_resolution"));
}