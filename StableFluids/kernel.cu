

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include "include/glad/glad.h"
#include "include/GLFW/glfw3.h"
#include "Shader.h"
#include "buffer.h"
#define uint unsigned int

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 1200;

#include "grid.h"
#include "linalg.h"
#include "cg.h"
#include "solve.h"
#include "advect.h"

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    _init_();
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "StableFluids", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    Shader vertex("Vertex.glsl", "Fragment.glsl");
    Grid gd = grid(DOMAIN_SIZE - 2);
    Buffer buff((void*)gd.vertices, (void*)gd.indices, gd.v_size, gd.i_size);
    GLuint ssbo; 
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    Mesh* reads = &read;
    Mesh* writes = &write;
    writes->n = reads->n;
    thread_setup(reads->n);
    dim3 fkernel_b(100, 100);
    dim3 fkernel_t(thread_dim, thread_dim);
    dim3 main_kernel_b(block_dim, block_dim);
    dim3 main_kernel_t(thread_dim, thread_dim);
    dim3 main_kernel_bb(block_dim - 2, block_dim - 2);
    LinearOperator_laplace(&mat, (reads->n - 2), reads->spacing);
    cudaMallocManaged(&linsys.q, sizeof(float) * (reads->n - 2 ) * (reads->n - 2));
    cudaMallocManaged(&linsys.r, sizeof(float) * (reads->n - 2)  * (reads->n - 2));
    cudaMallocManaged(&linsys.d, sizeof(float) * (reads->n - 2)  * (reads->n - 2));
    linsys.n_dim = (reads->n - 2) * (reads->n - 2);
    int cnt = 0;
    float max, min;
    max = 0.001f;
    min = -0.001f;
    glGenBuffers(1, &ssbo);
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {     
        if(cnt < 10)force <<<fkernel_b, 1 >>> (reads, 1.0f, .1f);    
        cudaDeviceSynchronize(); 
        cnt++;
        advect <<<main_kernel_bb, 1 >>> (reads, writes, 0.01f);
        cudaDeviceSynchronize();
        Mesh* ptr = reads;
        reads = writes;
        writes = ptr; 
        divergence <<< main_kernel_bb, 1 >> > (reads, reads->div, reads->spacing);
        cudaDeviceSynchronize();
        linsys.x = reads->p;     
        CG(&linsys, &mat, reads->div);   
        grad <<< main_kernel_bb, 1 >>> (reads, reads->spacing);
        cudaDeviceSynchronize();   
        add(&reads->u_x, &reads->u_x, &reads->grad_x, DOMAIN_SIZE * DOMAIN_SIZE, 1.0f);
        add(&reads->u_y, &reads->u_y, &reads->grad_y, DOMAIN_SIZE * DOMAIN_SIZE, 1.0f);  
        curl <<< main_kernel_bb, main_kernel_t >> > (reads, reads->curl, reads->spacing);
        cudaDeviceSynchronize(); 
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (DOMAIN_SIZE - 2) * (DOMAIN_SIZE - 2), read.p, GL_STATIC_DRAW); //sizeof(data) only works for statically sized C/C++ arrays.
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind     
        glUniform1f(3, min);
        glUniform1f(2, max);
        processInput(window);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(vertex.ID);
        glBindVertexArray(buff.VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        glDrawElements(GL_TRIANGLES,  6 * gd.v_size / sizeof(float), GL_UNSIGNED_INT, (void*)0);
        glfwSwapBuffers(window);
        glfwPollEvents();
        
    }
    glDeleteProgram(vertex.ID);
    glfwTerminate();
    _terminate_();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

