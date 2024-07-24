#pragma once
#define DOMAIN_SIZE 102
#define CELL_WIDTH  float a

struct Cell {
    float vx = 0;
    float vy = 0;
    float x = 0;
    float y = 0;
};
struct Mesh {
    int n;
    float* u_x, * u_y, * div, * p, * grad_x, * grad_y, * curl;
    float* coord;
    float spacing;
};

float* get_coordinates(float begin = 0, float end = 1.0) {
    auto data = (float*)malloc(sizeof(float) * (DOMAIN_SIZE + 1));
    float spacing =  (end - begin) / DOMAIN_SIZE;
    for (int i = 0; i < DOMAIN_SIZE; i++) {
        data[i] = begin;
        begin += spacing;
    }
    data[DOMAIN_SIZE] = spacing;
    CELL_WIDTH = spacing;
    std::cout << spacing << "\n";
    return data;
}