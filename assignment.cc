#include <dirent.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <jpeglib.h>

// interleaved RGB image struct RGB RGB RGB, row major:
// RGBRGBRGB
// RGBRGBRGB
// RGBRGBRGB
// above: example 3 x 3 image.
// 8 bits per channel.
struct Image {
  unsigned char* bytes;
  int width;
  int height;
};

struct Bone {
  int id;
  float length;
  glm::vec4 t;
  glm::vec4 n;
  glm::vec4 b;
  glm::vec4 first_world_point;
  glm::vec4 second_world_point;
  Bone* parent;
  std::vector<Bone*> children;
  glm::mat4 translate;
  glm::mat4 rotate;
  glm::mat4 deformed_rotate;
  glm::mat4 undeformed_to_world;
  glm::mat4 deformed_to_world;
};


struct BoneGraph {
  Bone *root;
};

struct BoneEndpoint {
    int id;
    int lineNum;
    int parentLineNum;
    glm::vec4 point;
};

std::vector<Bone*> bone_list(22);
BoneGraph b_graph;
std::vector<BoneEndpoint> bone_endpoints;
glm::vec4 eye_ray;
std::vector<std::vector<float> > skin_weights;
Bone* selected_bone;
std::vector<glm::vec4> cylinder_vertices;
std::vector<glm::uvec2> cylinder_faces;
std::vector<glm::vec4> origin_cylinder_vertices;
std::vector<glm::vec4> origin_ogre_vertices;


glm::vec4 center_of_mass;
float y_min;
float y_max;

std::vector<GLuint> textures;
GLuint sampler;
int curr_texture = 0;
GLint ogre_texture_location;
GLint ogre_texture_on_location;


int window_width = 800, window_height = 600;
const std::string window_title = "Virtual Mannequin";

const float kNear = 0.0001f;
const float kFar = 1000.0f;
const float kFov = 45.0f;
float aspect = static_cast<float>(window_width) / window_height;

// Floor info.
const float eps = 0.5 * (0.025 + 0.0175);
const float kFloorXMin = -100.0f;
const float kFloorXMax = 100.0f;
const float kFloorZMin = -100.0f;
const float kFloorZMax = 100.0f;
const float kFloorY = -0.75617 - eps;

const float kCylinderRadius = 0.05f;

enum {
  kMouseModeCamera,
  kMouseModeSkeleton,
  kNumMouseModes
};
int current_mouse_mode = 0;

// VBO and VAO descriptors.

// We have these VBOs available for each VAO.
enum {
  kVertexBuffer,
  kIndexBuffer,
  kNumVbos
};

// These are our VAOs.
enum {
  kFloorVao,
  kOgreVao,
  kBoneVao,
  kCylinderVao,
  kNumVaos
};

GLuint array_objects[kNumVaos];  // This will store the VAO descriptors.
GLuint buffer_objects[kNumVaos][kNumVbos];  // These will store VBO descriptors.

float last_x = 0.0f, last_y = 0.0f, current_x = 0.0f, current_y = 0.0f;
bool drag_state = false;
int current_button = -1;
float camera_distance = 2.0;
float pan_speed = 0.1f;
float roll_speed = 0.1f;
float rotation_speed = 0.05f;
float zoom_speed = 0.1f;
glm::vec3 eye = glm::vec3(0.0f, 0.1f, camera_distance);
glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 look = glm::vec3(0.0f, 0.0f, 1.0f);
glm::vec3 tangent = glm::cross(up, look);
glm::vec3 center = eye + camera_distance * look;
glm::mat3 orientation = glm::mat3(tangent, up, look);
bool fps_mode = false;

glm::mat4 view_matrix = glm::lookAt(eye, center, up);
glm::mat4 projection_matrix =
    glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect, kNear, kFar);
glm::mat4 model_matrix = glm::mat4(1.0f);
glm::mat4 floor_model_matrix = glm::mat4(1.0f);

const char* vertex_shader =
    "#version 330 core\n"
    "uniform vec4 light_position;"
    "in vec4 vertex_position;"
    "out vec4 vs_light_direction;"
    "void main() {"
    "gl_Position = vertex_position;"
    "vs_light_direction = light_position - gl_Position;"
    "}";

const char* geometry_shader =
    "#version 330 core\n"
    "layout (triangles) in;"
    "layout (triangle_strip, max_vertices = 3) out;"
    "uniform mat4 projection;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform vec4 light_position;"
    "in vec4 vs_light_direction[];"
    "out vec4 face_normal;"
    "out vec4 light_direction;"
    "out vec4 world_position;"
    "void main() {"
    "int n = 0;"
    "vec3 a = gl_in[0].gl_Position.xyz;"
    "vec3 b = gl_in[1].gl_Position.xyz;"
    "vec3 c = gl_in[2].gl_Position.xyz;"
    "vec3 u = normalize(b - a);"
    "vec3 v = normalize(c - a);"
    "face_normal = normalize(vec4(normalize(cross(u, v)), 0.0));"
    "for (n = 0; n < gl_in.length(); n++) {"
    "light_direction = normalize(vs_light_direction[n]);"
    "world_position = gl_in[n].gl_Position;"
    "gl_Position = projection * view * model * gl_in[n].gl_Position;"
    "EmitVertex();"
    "}"
    "EndPrimitive();"
    "}";

const char* bone_geometry_shader =
    "#version 330 core\n"
    "layout (lines) in;"
    "layout (line_strip, max_vertices = 2) out;"
    "uniform mat4 projection;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform vec4 selected;"
    "out vec4 world_position;"
    "out vec3 color;"
    "void main() {"
    "int n = 0;"
    "for (n = 0; n < gl_in.length(); n++) {"
    "world_position = gl_in[n].gl_Position;"
    "gl_Position = projection * view * model * gl_in[n].gl_Position;"
    "if(all(equal(gl_Position,selected))) {"
    "  color = vec3(1.0,0.0,0.0);"
    "}"
    "else {"
    "  color = vec3(0.0,1.0,0.0);"
    "}"
    "EmitVertex();"
    "}"
    "EndPrimitive();"
    "}";

const char* cylinder_geometry_shader =
    "#version 330 core\n"
    "layout (lines) in;"
    "layout (line_strip, max_vertices = 2) out;"
    "uniform mat4 projection;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "out vec4 world_position;"
    "void main() {"
    "int n = 0;"
    "vec3 a = gl_in[0].gl_Position.xyz;"
    "vec3 b = gl_in[1].gl_Position.xyz;"
    "for (n = 0; n < gl_in.length(); n++) {"
    "world_position = gl_in[n].gl_Position;"
    "gl_Position = projection * view * model * gl_in[n].gl_Position;"
    "EmitVertex();"
    "}"
    "EndPrimitive();"
    "}";

const char* floor_fragment_shader =
    "#version 330 core\n"
    "in vec4 face_normal;"
    "in vec4 light_direction;"
    "in vec4 world_position;"
    "out vec4 fragment_color;"
    "void main() {"
    "vec4 pos = world_position;"
    "float check_width = 0.25;"
    "float i = floor(pos.x / check_width);"
    "float j  = floor(pos.z / check_width);"
    "vec3 color = mod(i + j, 2) * vec3(1.0, 1.0, 1.0);"
    "float dot_nl = dot(normalize(light_direction), normalize(face_normal));"
    "dot_nl = clamp(dot_nl, 0.0, 1.0);"
    "color = clamp(dot_nl * color, 0.0, 1.0);"
    "fragment_color = vec4(color, 1.0);"
    "}";

const char* ogre_fragment_shader =
    "#version 330 core\n"
    "in vec4 face_normal;"
    "in vec4 light_direction;"
    "in vec4 world_position;"
    "out vec4 fragment_color;"
    "uniform vec4 center_of_mass;"
    "uniform sampler2D texture_sampler;"
    "uniform float y_min;"
    "uniform float y_max;"
    "uniform int is_texture_on;"
    "void main() {"
    "vec4 color;"
    "if(is_texture_on == 1){"
    "float u = (atan(world_position.x - center_of_mass.x, world_position.z - center_of_mass.z) + 3.14159265)/(2*3.14159265);"
    "float v = (world_position.y - y_min)/(y_max - y_min);"
    "vec2 tex_coord = vec2(u,v);"
    "color = texture2D(texture_sampler, tex_coord);"
    "}"
    "else {"
    "color = vec4(0.0, 1.0, 0.0, .5);"
    "}"
    "float dot_nl = dot(normalize(light_direction), normalize(face_normal));"
    "dot_nl = clamp(dot_nl, 0.0, 1.0);"
    "color = clamp(dot_nl * color, 0.0, 1.0);"
    "fragment_color = color;"
    "}";

const char* bone_fragment_shader = 
    "#version 330 core\n"
    "in vec3 color;"
    "out vec4 fragment_color;"
    "void main() {"
    "fragment_color = vec4(color, .5);"
    "}";

const char* cylinder_fragment_shader = 
    "#version 330 core\n"
    "in vec3 color;"
    "uniform vec3 line_color;"
    "out vec4 fragment_color;"
    "void main() {"
    "fragment_color = vec4(line_color, 1.0);"
    "}";

const char* OpenGlErrorToString(GLenum error) {
  switch (error) {
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
      break;
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
      break;
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
      break;
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
      break;
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
      break;
    default:
      return "Unknown Error";
      break;
  }
  return "Unicorns Exist";
}

#define CHECK_SUCCESS(x) \
  if (!(x)) {            \
    glfwTerminate();     \
    exit(EXIT_FAILURE);  \
  }

#define CHECK_GL_SHADER_ERROR(id)                                           \
  {                                                                         \
    GLint status = 0;                                                       \
    GLint length = 0;                                                       \
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);                          \
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);                         \
    if (!status) {                                                          \
      std::string log(length, 0);                                           \
      glGetShaderInfoLog(id, length, nullptr, &log[0]);                     \
      std::cerr << "Line :" << __LINE__ << " OpenGL Shader Error: Log = \n" \
                << &log[0];                                                 \
      glfwTerminate();                                                      \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  }

#define CHECK_GL_PROGRAM_ERROR(id)                                           \
  {                                                                          \
    GLint status = 0;                                                        \
    GLint length = 0;                                                        \
    glGetProgramiv(id, GL_LINK_STATUS, &status);                             \
    glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);                         \
    if (!status) {                                                           \
      std::string log(length, 0);                                            \
      glGetProgramInfoLog(id, length, nullptr, &log[0]);                     \
      std::cerr << "Line :" << __LINE__ << " OpenGL Program Error: Log = \n" \
                << &log[0];                                                  \
      glfwTerminate();                                                       \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }

#define CHECK_GL_ERROR(statement)                                             \
  {                                                                           \
    { statement; }                                                            \
    GLenum error = GL_NO_ERROR;                                               \
    if ((error = glGetError()) != GL_NO_ERROR) {                              \
      std::cerr << "Line :" << __LINE__ << " OpenGL Error: code  = " << error \
                << " description =  " << OpenGlErrorToString(error);          \
      glfwTerminate();                                                        \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  size_t count = std::min(v.size(), static_cast<size_t>(10));
  for (size_t i = 0; i < count; ++i) os << i << " " << v[i] << "\n";
  os << "size = " << v.size() << "\n";
  return os;
}

namespace glm {
  std::ostream& operator<<(std::ostream& os, const glm::vec2& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::vec3& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::mat4& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::mat3& v) {
    os << glm::to_string(v);
    return os;
  }
}  // namespace glm


// Bone* Search_Bone_Graph(Bone *curr_bone, int id) {
//   if(!curr_bone->children.empty()) {
//     for(int i = 0; i < curr_bone->children.size(); ++i)
//       Search_Scene_Graph<T>(curr_bone->children[i], nodes);
//   }
//   if(T* node = dynamic_cast<T*>(curr_bone))
//     nodes.push_back(node);
// }

void PrintBoneGraph(const BoneGraph& bg, Bone* curr_bone) {
  for(auto child : curr_bone->children) {
    PrintBoneGraph(bg, child);
  }
  std::cout << "Bone: " << curr_bone->id << " rotate: " << curr_bone->rotate << std::endl;
}

void LoadObj(const std::string& file, std::vector<glm::vec4>& vertices, std::vector<glm::uvec3>& indices) {
  std::ifstream in(file);
  int i = 0, j = 0;
  glm::vec4 vertex = glm::vec4(0.0, 0.0, 0.0, 1.0);
  glm::uvec3 face_indices = glm::uvec3(0, 0, 0);
  while (in.good()) {
    char c = in.get();
    switch (c) {
      case 'v':
        in >> vertex[0] >> vertex[1] >> vertex[2];
        vertices.push_back(vertex);
        break;
      case 'f':
        in >> face_indices[0] >> face_indices[1] >> face_indices[2];
        face_indices -= 1;
        indices.push_back(face_indices);
        break;
      default:
        break;
    }
  }
  in.close();
}

void LoadSkinWeights(const std::string& file) {
  std::ifstream in(file);
  int num_bones, num_verts;
  float weight;
  in >> num_bones >> num_verts;
  for(int i = 0; i < num_bones; ++i) {
    std::vector<float> verts_for_bone;
    for(int j = 0; j < num_verts; ++j) {
      in >> weight;
      verts_for_bone.push_back(weight);
    }
    skin_weights.push_back(verts_for_bone);
  }
  in.close();
}

void LoadTextures(std::vector<Image>& images) {
  y_min = origin_ogre_vertices[0][1];
  y_max = origin_ogre_vertices[0][1];
  center_of_mass = glm::vec4(0,0,0,0);
  for(int i = 0; i < origin_ogre_vertices.size(); ++i) {
    center_of_mass += origin_ogre_vertices[i];
    if(origin_ogre_vertices[i][1] > y_max)
      y_max = origin_ogre_vertices[i][1];
    else if(origin_ogre_vertices[i][1] < y_min)
      y_min = origin_ogre_vertices[i][1];
  }
  center_of_mass /= origin_ogre_vertices.size();


  CHECK_GL_ERROR(glGenTextures(images.size(), &textures[0]));
  CHECK_GL_ERROR(glGenSamplers(1, &sampler));
  CHECK_GL_ERROR(glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_REPEAT));
  CHECK_GL_ERROR(glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_REPEAT));
  CHECK_GL_ERROR(glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  CHECK_GL_ERROR(glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  for(int i = 0; i < images.size(); ++i) {
    CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, textures[i]));
    CHECK_GL_ERROR(glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, images[i].width, images[i].height));
    CHECK_GL_ERROR(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, images[i].width, images[i].height, GL_RGB, GL_UNSIGNED_BYTE, images[i].bytes)); 
  }



}


void LoadBonesRecursive(Bone* curr_bone, glm::mat4 rot) {

  if(curr_bone->id != -1) {
    glm::vec4 t = bone_endpoints[curr_bone->id+1].point;

    std::cout << "\n\nlength value for id: " << curr_bone->id << " is " << curr_bone->length << std::endl;
    std::cout << "t value for id: " << curr_bone->id << " is " << t << " and it's length is : " << glm::length(bone_endpoints[curr_bone->id+1].point) << std::endl;
    // std::cout << "rot matrix: " << rot << std::endl;
    curr_bone->translate = glm::transpose(glm::mat4(1.000000, 0.000000, 0.000000, 0,
                                                    0.000000, 1.000000, 0.000000, curr_bone->length,
                                                    0.000000, 0.000000, 1.000000, 0,
                                                    0.000000, 0.000000, 0.000000, 1.000000));
    curr_bone->t = t;
    t = glm::vec4(glm::normalize(glm::vec3(glm::inverse(rot)*t)), 0);
    // std::cout << "t is " << t << std::endl;
    
    glm::vec4 n;
    float x = abs(bone_endpoints[curr_bone->id+1].point[0]);
    float y = abs(bone_endpoints[curr_bone->id+1].point[1]);
    float z = abs(bone_endpoints[curr_bone->id+1].point[2]);
    if(x == std::min(std::min(x,y),z)){
      n = glm::vec4(1,0,0,0);
    }
    else if(y == std::min(std::min(x,y),z)){
      n = glm::vec4(0,1,0,0);
    }
    else {
      n = glm::vec4(0,0,1,0);
    }

    n = glm::vec4(glm::normalize(glm::cross(glm::vec3(t),glm::vec3(n))), 0);
    curr_bone->n = n;
    glm::vec4 b = glm::vec4(glm::normalize(glm::cross(glm::vec3(t),glm::vec3(n))), 0); 
    curr_bone->b = b;

    curr_bone->rotate = glm::transpose(glm::mat4( n[0], t[0], b[0], 0,
                                                  n[1], t[1], b[1], 0,
                                                  n[2], t[2], b[2], 0,
                                                  0.00, 0.00, 0.00, 1.00));
    curr_bone->deformed_rotate = curr_bone->rotate; 
     std::cout << "curr_bone->rotate " << curr_bone->rotate << "at " << curr_bone->id << std::endl;
  }
  for(int j = 0; j < bone_endpoints.size(); ++j) {
    // find all lines in file for which the parentLineNum refers to curr_bone
    if(bone_endpoints[j].parentLineNum == curr_bone->id + 1) { // id + 1 may not always be correct
      Bone* child = new Bone();
      child->id = bone_endpoints[j].id;
      child->length = glm::length(bone_endpoints[j].point);
      child->parent = curr_bone;
      curr_bone->children.push_back(child);
      bone_list[child->id] = child;
      LoadBonesRecursive(child, rot*curr_bone->rotate);
    }
  }

}

void LoadBones(const std::string& file, BoneGraph& b_graph) {
  std::ifstream in(file);
  int lineNum = 0; 
  Bone* root = new Bone();
  int id,parent;
  float x,y,z;
  while(!(in >> id).eof()) {
    in >> parent;
    in >> x >> y >> z;
    BoneEndpoint endpoint;
    endpoint.id = id;
    endpoint.lineNum = lineNum;
    endpoint.parentLineNum = parent;
    endpoint.point = glm::vec4(x,y,z,0);
    bone_endpoints.push_back(endpoint);
    if(id == -1)
    {
        root->translate = glm::transpose(glm::mat4(1.000000, 0.000000, 0.000000, x,
                                                    0.000000, 1.000000, 0.000000, y,
                                                    0.000000, 0.000000, 1.000000, z,
                                                    0.000000, 0.000000, 0.000000, 1.000000));
    }

    ++lineNum;
  }
  
  root->length = 0;
  root->id = -1;
  root->rotate = glm::mat4(1);
  bone_list[root->id] = root;
  LoadBonesRecursive(root, root->rotate);
  b_graph.root = root;
  in.close();
}

void LoadBoneLines(Bone* curr_bone, std::vector<glm::vec4>& vertices,
             std::vector<glm::uvec2>& lines, glm::mat4 m, glm::vec4 firstPoint)
{
  vertices.push_back(firstPoint);
  curr_bone->first_world_point = firstPoint;
  glm::mat4 m_new = m * curr_bone->deformed_rotate * curr_bone->translate;
  glm::vec4 secondPoint = m_new * glm::vec4(0,0,0,1);
  curr_bone->second_world_point = secondPoint;
  vertices.push_back(secondPoint);
  lines.push_back(glm::uvec2(vertices.size()-2, vertices.size()-1));

  for(auto child : curr_bone->children) {
    LoadBoneLines(child, vertices, lines, m_new, secondPoint);
  }
}

glm::mat4 getWorldTransform(Bone* b) {
  glm::mat4 transform = b->rotate;
  Bone* curr_bone = b->parent;
  while(curr_bone)
  {
    transform = curr_bone->rotate * curr_bone->translate * transform;
    curr_bone = curr_bone->parent;
  }
  return transform;
}

glm::mat4 getDeformedTransform(Bone* b) {
  glm::mat4 transform = b->deformed_rotate;
  Bone* curr_bone = b->parent;
  while(curr_bone)
  {
    transform = curr_bone->deformed_rotate * curr_bone->translate * transform;
    curr_bone = curr_bone->parent;
  }
  return transform;
}
  // glm::mat4 transform = glm::transpose(glm::mat4(1.000000, 0.000000, 0.000000, b->first_world_point[0],
  //                                                   0.000000, 1.000000, 0.000000, b->first_world_point[1],
  //                                                   0.000000, 0.000000, 1.000000, b->first_world_point[2],
  //                                                   0.000000, 0.000000, 0.000000, 1.000000));
  // glm::vec4 t = b->second_world_point - b->first_world_point;
  // glm::vec4 n;
  // float x = abs(bone_endpoints[b->id+1].point[0]);
  // float y = abs(bone_endpoints[b->id+1].point[1]);
  // float z = abs(bone_endpoints[b->id+1].point[2]);
  // if(x == std::min(std::min(x,y),z)){
  //   n = glm::vec4(1,0,0,0);
  // }
  // else if(y == std::min(std::min(x,y),z)){
  //   n = glm::vec4(0,1,0,0);
  // }
  // else {
  //   n = glm::vec4(0,0,1,0);
  // }

  // n = glm::vec4(glm::normalize(glm::cross(glm::vec3(t),glm::vec3(n))), 0);
  // glm::vec4 b_dir = glm::vec4(glm::normalize(glm::cross(glm::vec3(t),glm::vec3(n))), 0); 

  // transform *= glm::transpose(glm::mat4( n[0], t[0], b_dir[0], 0,
  //                                                 n[1], t[1], b_dir[1], 0,
  //                                                 n[2], t[2], b_dir[2], 0,
  //                                                 0.00, 0.00, 0.00, 1.00));


  
  // return transform;
// }

Bone* Perform_Ray_Collision() {
  //std::map<std::pair<int,Bone*> >collided_bones; // int = t value at which the ray collided
  glm::mat4 world_transform;
  glm::vec4 bspace_eye_ray,bspace_eye_origin;
  glm::vec4 projected_eye_ray,projected_eye_origin;
  glm::vec4 collided_point;

  float b,c,t=-1;
  float min_t = kFar;

  Bone* closest_bone = NULL;

  for(int i = 0; i < bone_list.size(); ++i)
  {
    Bone* curr_bone = bone_list[i];
    world_transform = glm::inverse(getDeformedTransform(curr_bone));
    // world_transform = glm::mat4(1);
    bspace_eye_ray = glm::normalize(world_transform * eye_ray);
    bspace_eye_origin = world_transform * glm::vec4(eye,0);
    projected_eye_ray = glm::normalize(glm::vec4(bspace_eye_ray[0], 0, bspace_eye_ray[2], 0));
    projected_eye_origin = glm::vec4(bspace_eye_origin[0], 0, bspace_eye_origin[2], 0); // turned into a vector for easy math
    b = 2*glm::dot(projected_eye_origin, projected_eye_ray);
    c = glm::dot(projected_eye_origin,projected_eye_origin) - kCylinderRadius*kCylinderRadius;
    if(b*b - 4*c >= 0) {
      if((-b - sqrt(b*b - 4*c))/2 > 0)
        t = (-b - sqrt(b*b - 4*c))/2;
      else
        t = (-b + sqrt(b*b - 4*c))/2;
    }

    collided_point = bspace_eye_origin + t*bspace_eye_ray;
    if(t>0 && collided_point[1] < curr_bone->length && collided_point[1] > 0)
    {
      if(t < min_t) {
        closest_bone = curr_bone;
        min_t = t;
      }
    }
  }
  
  return closest_bone;
}

void RunLinearSkinning(std::vector<glm::vec4> &ogre_verts) {
  for(int i = 0; i < ogre_verts.size(); ++i)
    ogre_verts[i] = glm::vec4(0,0,0,0);

  for(int i = 0; i < bone_list.size(); ++i) {
    Bone* curr_bone = bone_list[i];
    glm::mat4 Ui = glm::inverse(getWorldTransform(curr_bone));
    glm::mat4 Di = glm::inverse(getDeformedTransform(curr_bone));
    for(int j = 0; j < ogre_verts.size(); ++j)
      ogre_verts[j] += skin_weights[i][j] * glm::inverse(Di) * Ui * origin_ogre_vertices[j];
  }


}
void SaveJPEG(const std::string& filename, int image_width, int image_height, const unsigned char* pixels) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE* outfile;
  JSAMPROW row_pointer[1];
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  CHECK_SUCCESS((outfile = fopen(filename.c_str(), "wb")) != NULL)

  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = image_width;
  cinfo.image_height = image_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, true);
  jpeg_start_compress(&cinfo, true);

  row_stride = image_width * 3;

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = const_cast<unsigned char*>(
        &pixels[(cinfo.image_height - 1 - cinfo.next_scanline) * row_stride]);
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);

  jpeg_destroy_compress(&cinfo);
}

bool LoadJPEG(const std::string& file_name, Image* image) {
  FILE* file = fopen(file_name.c_str(), "rb");
  struct jpeg_decompress_struct info;
  struct jpeg_error_mgr err;

  info.err = jpeg_std_error(&err);
  jpeg_create_decompress(&info);

  CHECK_SUCCESS(file != NULL);

  jpeg_stdio_src(&info, file);
  jpeg_read_header(&info, true);
  jpeg_start_decompress(&info);

  image->width = info.output_width;
  image->height = info.output_height;

  int channels = info.num_components;
  long size = image->width * image->height * 3;

  image->bytes = new unsigned char[size];

  int a = (channels > 2 ? 1 : 0);
  int b = (channels > 2 ? 2 : 0);
  std::vector<unsigned char> scan_line(image->width * channels, 0);
  unsigned char* p1 = &scan_line[0];
  unsigned char** p2 = &p1;
  unsigned char* out_scan_line = &image->bytes[0];
  while (info.output_scanline < info.output_height) {
    jpeg_read_scanlines(&info, p2, 1);
    for (int i = 0; i < image->width; ++i) {
      out_scan_line[3 * i] = scan_line[channels * i];
      out_scan_line[3 * i + 1] = scan_line[channels * i + a];
      out_scan_line[3 * i + 2] = scan_line[channels * i + b];
    }
    out_scan_line += image->width * 3;
  }
  jpeg_finish_decompress(&info);
  fclose(file);
  return true;
}

void ErrorCallback(int error, const char* description) {
  std::cerr << "GLFW Error: " << description << "\n";
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  else if (key == GLFW_KEY_W && action != GLFW_RELEASE) {
    if (fps_mode)
      eye -= zoom_speed * look;
    else
      camera_distance -= zoom_speed;
  } else if (key == GLFW_KEY_S && action != GLFW_RELEASE) {
    if (fps_mode)
      eye += zoom_speed * look;
    else
      camera_distance += zoom_speed;
  } else if (key == GLFW_KEY_A && action != GLFW_RELEASE) {
    if (fps_mode)
      eye -= pan_speed * tangent;
    else
      center -= pan_speed * tangent;
  } else if (key == GLFW_KEY_D && action != GLFW_RELEASE) {
    if (fps_mode)
      eye += pan_speed * tangent;
    else
      center += pan_speed * tangent;
  } else if (key == GLFW_KEY_LEFT && action != GLFW_RELEASE) {
    if (current_mouse_mode == kMouseModeCamera) {
      glm::mat3 rotation = glm::mat3(glm::rotate(-roll_speed, look));
      orientation = rotation * orientation;
      tangent = glm::column(orientation, 0);
      up = glm::column(orientation, 1);
      look = glm::column(orientation, 2);
    } else {
      if(selected_bone) {
        glm::vec4 axis = glm::normalize(selected_bone->deformed_rotate * glm::vec4(0,1,0,0));
        selected_bone->deformed_rotate = glm::rotate(-rotation_speed, glm::vec3(axis)) * selected_bone->deformed_rotate;
      }
    }
  } else if (key == GLFW_KEY_RIGHT && action != GLFW_RELEASE) {
    if (current_mouse_mode == kMouseModeCamera) {
      glm::mat3 rotation = glm::mat3(glm::rotate(roll_speed, look));
      orientation = rotation * orientation;
      tangent = glm::column(orientation, 0);
      up = glm::column(orientation, 1);
      look = glm::column(orientation, 2);
    } else {
      if(selected_bone) {
        glm::vec4 axis = glm::normalize(selected_bone->deformed_rotate * glm::vec4(0,1,0,0));
        selected_bone->deformed_rotate = glm::rotate(rotation_speed, glm::vec3(axis)) * selected_bone->deformed_rotate;
      }
    }
  } else if (key == GLFW_KEY_DOWN && action != GLFW_RELEASE) {
    if (fps_mode)
      eye -= pan_speed * up;
    else
      center -= pan_speed * up;
  } else if (key == GLFW_KEY_UP && action != GLFW_RELEASE) {
    if (fps_mode)
      eye += pan_speed * up;
    else
      center += pan_speed * up;
  } else if (key == GLFW_KEY_C && action != GLFW_RELEASE) {
    fps_mode = !fps_mode;
  } else if (key == GLFW_KEY_M && action != GLFW_RELEASE) {
    current_mouse_mode = (current_mouse_mode + 1) % kNumMouseModes;
  } else if (key == GLFW_KEY_J && action != GLFW_RELEASE) {
    std::vector<unsigned char> pixels(3 * window_width * window_height, 0);
    CHECK_GL_ERROR(glReadPixels(0, 0, window_width, window_height, GL_RGB,
                                GL_UNSIGNED_BYTE, &pixels[0]));
    std::string filename = "capture.jpg";
    std::cout << "Encoding and saving to file '" + filename + "'\n";
    SaveJPEG(filename, window_width, window_height, &pixels[0]);
  } else if (key == GLFW_KEY_T && action != GLFW_RELEASE) {
    if(curr_texture < textures.size()) {
      CHECK_GL_ERROR(glUniform1i(ogre_texture_on_location, 1)); // turn on textures
      std::cout << "Turned on textures" << std::endl << std::flush;

      CHECK_GL_ERROR(glActiveTexture(GL_TEXTURE0));
      std::cout << "Set Active Texture" << std::endl << std::flush;
      CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, textures[curr_texture]));
      std::cout << "Bound Texture" << std::endl << std::flush;
      CHECK_GL_ERROR(glBindSampler(textures[curr_texture], sampler));
      std::cout << "Bound Sampler" << std::endl << std::flush;
      CHECK_GL_ERROR(glUniform1i(ogre_texture_location, 0));
      std::cout << "Passed Uniform" << std::endl << std::flush;
      ++curr_texture;
    }
    else {
      curr_texture = 0;
      CHECK_GL_ERROR(glUniform1i(ogre_texture_on_location, 0)); // turn off textures
    }
  }
}

void MousePosCallback(GLFWwindow* window, double mouse_x, double mouse_y) {
  last_x = current_x;
  last_y = current_y;
  current_x = mouse_x;
  current_y = window_height - mouse_y;
  float delta_x = current_x - last_x;
  float delta_y = current_y - last_y;
  if (sqrt(delta_x * delta_x + delta_y * delta_y) < 1e-15) return;
  glm::vec3 mouse_direction = glm::normalize(glm::vec3(delta_x, delta_y, 0.0f));
  glm::vec2 mouse_start = glm::vec2(last_x, last_y);
  glm::vec2 mouse_end = glm::vec2(current_x, current_y);
  glm::uvec4 viewport = glm::uvec4(0, 0, window_width, window_height);
  if (drag_state && current_button == GLFW_MOUSE_BUTTON_LEFT) {
    if (current_mouse_mode == kMouseModeCamera) {
      glm::vec3 axis = glm::normalize(
          orientation * glm::vec3(mouse_direction.y, -mouse_direction.x, 0.0f));
      orientation =
          glm::mat3(glm::rotate(rotation_speed, axis) * glm::mat4(orientation));
      tangent = glm::column(orientation, 0);
      up = glm::column(orientation, 1);
      look = glm::column(orientation, 2);
    }
    else {
      if(selected_bone) {
        glm::vec4 axis = glm::normalize(selected_bone->deformed_rotate * glm::vec4(mouse_direction.y, 0, -mouse_direction.x, 0.0f));
        selected_bone->deformed_rotate = glm::rotate(rotation_speed, glm::vec3(axis)) * selected_bone->deformed_rotate;
      }
    }
  }
  else {
    std::vector<float> depth_buffer(window_height * window_width, 0);
    CHECK_GL_ERROR(glReadPixels(0, 0, window_width, window_height, GL_DEPTH_COMPONENT, GL_FLOAT, &depth_buffer[0]));
    float x = 2*mouse_x/window_width - 1;
    float y = 1 - 2*mouse_y/window_height;
    int depth_index = (int)(mouse_x + (window_height - mouse_y) * window_width);
    float z = -1;
    glm::vec4 pos = glm::vec4(x,y,z,1);
    pos = glm::inverse(projection_matrix) * pos;
    pos[0] /= pos[3];
    pos[1] /= pos[3];
    pos[2] /= pos[3];
    pos[3] = 1;
    pos = glm::inverse(view_matrix) * pos;
    pos[0] /= pos[3];
    pos[1] /= pos[3];
    pos[2] /= pos[3];
    pos[3] = 1;
    //eye_ray = glm::vec4(eye_ray.x, eye_ray.y, eye, 0.0);

    eye_ray = pos - glm::vec4(eye,1);
    eye_ray = glm::normalize(eye_ray);

   // glm::vec3 edir = glm::cross(glm::vec3(eye_ray[0],eye_ray[1],eye_ray[2]), up);
   // eye_ray = glm::vec4(edir,0);  
    // maybe put some code here
  }
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  drag_state = (action == GLFW_PRESS);
  current_button = button;
}

int main(int argc, char* argv[]) {
  if (!glfwInit()) exit(EXIT_FAILURE);
  glfwSetErrorCallback(ErrorCallback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  GLFWwindow* window = glfwCreateWindow(window_width, window_height,
                                        &window_title[0], nullptr, nullptr);
  CHECK_SUCCESS(window != nullptr);

  glfwMakeContextCurrent(window);
  glewExperimental = GL_TRUE;
  CHECK_SUCCESS(glewInit() == GLEW_OK);
  glGetError();  // clear GLEW's error for it

  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCursorPosCallback(window, MousePosCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);
  glfwSwapInterval(1);
  const GLubyte* renderer = glGetString(GL_RENDERER);  // get renderer string
  const GLubyte* version = glGetString(GL_VERSION);    // version as a string
  std::cout << "Renderer: " << renderer << "\n";
  std::cout << "OpenGL version supported:" << version << "\n";

  std::vector<glm::vec4> floor_vertices;
  std::vector<glm::uvec3> floor_faces;
  floor_vertices.push_back(glm::vec4(kFloorXMin, kFloorY, kFloorZMax, 1.0f));
  floor_vertices.push_back(glm::vec4(kFloorXMax, kFloorY, kFloorZMax, 1.0f));
  floor_vertices.push_back(glm::vec4(kFloorXMax, kFloorY, kFloorZMin, 1.0f));
  floor_vertices.push_back(glm::vec4(kFloorXMin, kFloorY, kFloorZMin, 1.0f));
  floor_faces.push_back(glm::uvec3(0, 1, 2));
  floor_faces.push_back(glm::uvec3(2, 3, 0));

  std::vector<std::string> jpeg_file_names;
  DIR* dir;
  struct dirent* entry;
  CHECK_SUCCESS((dir = opendir("./textures")) != NULL);
  while ((entry = readdir(dir)) != NULL) {
    std::string file_name(entry->d_name);
    std::transform(file_name.begin(), file_name.end(), file_name.begin(),
                   tolower);
    if (file_name.find(".jpg") != std::string::npos) {
      jpeg_file_names.push_back(file_name);
    }
  }
  closedir(dir);

  std::vector<Image> images(jpeg_file_names.size());
  for (int i = 0; i < jpeg_file_names.size(); ++i) {
    std::string file_name = "./textures/" + jpeg_file_names[i];
    LoadJPEG(file_name, &images[i]);
    std::cout << "Loaded '" << file_name << "' width = " << images[i].width
              << " height = " << images[i].height << "\n";
  }

  std::vector<glm::vec4> ogre_vertices;
  std::vector<glm::uvec3> ogre_faces;

  LoadObj("ogre-rigged/ogre.obj", ogre_vertices, ogre_faces);
  LoadBones("ogre-rigged/ogre-skeleton.bf", b_graph);
  LoadSkinWeights("ogre-rigged/ogre-weights.dmat");

  origin_ogre_vertices = ogre_vertices;

  textures.resize(images.size()); 
  LoadTextures(images);

  

  //PrintBoneGraph(b_graph, b_graph.root);
  std::vector<glm::vec4> bone_vertices;
  std::vector<glm::uvec2> bone_lines;
  std::cout << "Starting LoadBoneLines\n";
  for(auto child : b_graph.root->children) {
    LoadBoneLines(child, bone_vertices, bone_lines, b_graph.root->translate,b_graph.root->translate * glm::vec4(0,0,0,1));
  }
  glm::vec4 test = glm::vec4(0, 0, 0, 1);
  glm::vec4 test2 = glm::vec4(0, 0, -1, 1);
  bone_vertices.push_back(test);
  bone_vertices.push_back(test2);
  bone_lines.push_back(glm::uvec2(bone_vertices.size()-2, bone_vertices.size()-1));
  // for(auto vertex : bone_vertices)
  //   std::cout << "Vertex: " << vertex << std::endl;

  Bone* s_bone = b_graph.root->children[0];
  for (int i = 0; i <= 15; ++i) {
      
      float angle = i*((1.0/15) * (2*3.14));
      //glNormal3d(cos(angle),sin(angle),0);
      cylinder_vertices.push_back(glm::vec4(kCylinderRadius*cos(angle), 1, kCylinderRadius*sin(angle), 1));
      cylinder_vertices.push_back(glm::vec4(kCylinderRadius*cos(angle), .5, kCylinderRadius*sin(angle), 1));
      cylinder_vertices.push_back(glm::vec4(kCylinderRadius*cos(angle), 0, kCylinderRadius*sin(angle), 1));  
      cylinder_faces.push_back(glm::uvec2(cylinder_vertices.size()-3, cylinder_vertices.size()-1));
      if(i>0)
      {
        cylinder_faces.push_back(glm::uvec2(cylinder_vertices.size()-4, cylinder_vertices.size()-1));
        cylinder_faces.push_back(glm::uvec2(cylinder_vertices.size()-5, cylinder_vertices.size()-2));
        cylinder_faces.push_back(glm::uvec2(cylinder_vertices.size()-6, cylinder_vertices.size()-3));
      }
  }
  cylinder_vertices.push_back(glm::vec4(0,0,0,1));
  cylinder_vertices.push_back(glm::vec4(0,0,2 * kCylinderRadius,1));
  cylinder_faces.push_back(glm::uvec2(cylinder_vertices.size()-2, cylinder_vertices.size()-1));
  cylinder_vertices.push_back(glm::vec4(2 * kCylinderRadius,0,0,1));
  cylinder_faces.push_back(glm::uvec2(cylinder_vertices.size()-3, cylinder_vertices.size()-1));

  origin_cylinder_vertices = cylinder_vertices;

  // Setup our VAOs.
  CHECK_GL_ERROR(glGenVertexArrays(kNumVaos, array_objects));

  // Setup the object array object.

  // Switch to the floor VAO.
  CHECK_GL_ERROR(glBindVertexArray(array_objects[kFloorVao]));

  // Generate buffer objects
  CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kFloorVao][0]));

  // Setup vertex data in a VBO.
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kFloorVao][kVertexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * floor_vertices.size() * 4,
                              &floor_vertices[0], GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  // Setup element array buffer.
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              buffer_objects[kFloorVao][kIndexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              sizeof(uint32_t) * floor_faces.size() * 3,
                              &floor_faces[0], GL_STATIC_DRAW));

  // Switch to the ogre VAO.
  CHECK_GL_ERROR(glBindVertexArray(array_objects[kOgreVao]));

  // Generate buffer objects
  CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kOgreVao][0]));

  // Setup vertex data in a VBO.
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kOgreVao][kVertexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * ogre_vertices.size() * 4,
                              &ogre_vertices[0], GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  // Setup element array buffer.
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              buffer_objects[kOgreVao][kIndexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              sizeof(uint32_t) * ogre_faces.size() * 3,
                              &ogre_faces[0], GL_STATIC_DRAW));



      // Switch to the cylinder VAO.
  CHECK_GL_ERROR(glBindVertexArray(array_objects[kCylinderVao]));

  // Generate buffer objects
  CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kCylinderVao][0]));

  // Setup vertex data in a VBO.
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kCylinderVao][kVertexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * cylinder_vertices.size() * 4,
                              &cylinder_vertices[0], GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  // Setup element array buffer.
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              buffer_objects[kCylinderVao][kIndexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              sizeof(uint32_t) * cylinder_faces.size() * 2,
                              &cylinder_faces[0], GL_STATIC_DRAW));

  // Triangle shaders

  // Setup vertex shader.
  GLuint vertex_shader_id = 0;
  const char* vertex_source_pointer = vertex_shader;
  CHECK_GL_ERROR(vertex_shader_id = glCreateShader(GL_VERTEX_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(vertex_shader_id, 1, &vertex_source_pointer, nullptr));
  glCompileShader(vertex_shader_id);
  CHECK_GL_SHADER_ERROR(vertex_shader_id);

  // Setup geometry shader.
  GLuint geometry_shader_id = 0;
  const char* geometry_source_pointer = geometry_shader;
  CHECK_GL_ERROR(geometry_shader_id = glCreateShader(GL_GEOMETRY_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(geometry_shader_id, 1, &geometry_source_pointer, nullptr));
  glCompileShader(geometry_shader_id);
  CHECK_GL_SHADER_ERROR(geometry_shader_id);

  // Setup bone geometry shader.
  GLuint bone_geometry_shader_id = 1;
  const char* bone_geometry_source_pointer = bone_geometry_shader;
  CHECK_GL_ERROR(bone_geometry_shader_id = glCreateShader(GL_GEOMETRY_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(bone_geometry_shader_id, 1, &bone_geometry_source_pointer, nullptr));
  glCompileShader(bone_geometry_shader_id);
  CHECK_GL_SHADER_ERROR(bone_geometry_shader_id);

    // Setup cylinder geometry shader.
  GLuint cylinder_geometry_shader_id = 2;
  const char* cylinder_geometry_source_pointer = cylinder_geometry_shader;
  CHECK_GL_ERROR(cylinder_geometry_shader_id = glCreateShader(GL_GEOMETRY_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(cylinder_geometry_shader_id, 1, &cylinder_geometry_source_pointer, nullptr));
  glCompileShader(cylinder_geometry_shader_id);
  CHECK_GL_SHADER_ERROR(cylinder_geometry_shader_id);

  // Setup floor fragment shader.
  GLuint floor_fragment_shader_id = 0;
  const char* floor_fragment_source_pointer = floor_fragment_shader;
  CHECK_GL_ERROR(floor_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER));
  CHECK_GL_ERROR(glShaderSource(floor_fragment_shader_id, 1,
                                &floor_fragment_source_pointer, nullptr));
  glCompileShader(floor_fragment_shader_id);
  CHECK_GL_SHADER_ERROR(floor_fragment_shader_id);

  // Setup ogre fragment shader.
  GLuint ogre_fragment_shader_id = 1;
  const char* ogre_fragment_source_pointer = ogre_fragment_shader;
  CHECK_GL_ERROR(ogre_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER));
  CHECK_GL_ERROR(glShaderSource(ogre_fragment_shader_id, 1,
                                &ogre_fragment_source_pointer, nullptr));
  glCompileShader(ogre_fragment_shader_id);
  CHECK_GL_SHADER_ERROR(ogre_fragment_shader_id);

  // Setup bone fragment shader.
  GLuint bone_fragment_shader_id = 2;
  const char* bone_fragment_source_pointer = bone_fragment_shader;
  CHECK_GL_ERROR(bone_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER));
  CHECK_GL_ERROR(glShaderSource(bone_fragment_shader_id, 1,
                                &bone_fragment_source_pointer, nullptr));
  glCompileShader(bone_fragment_shader_id);
  CHECK_GL_SHADER_ERROR(bone_fragment_shader_id);

    // Setup cylinder fragment shader.
  GLuint cylinder_fragment_shader_id = 3;
  const char* cylinder_fragment_source_pointer = cylinder_fragment_shader;
  CHECK_GL_ERROR(cylinder_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER));
  CHECK_GL_ERROR(glShaderSource(cylinder_fragment_shader_id, 1,
                                &cylinder_fragment_source_pointer, nullptr));
  glCompileShader(cylinder_fragment_shader_id);
  CHECK_GL_SHADER_ERROR(cylinder_fragment_shader_id);

  // Let's create our floor program.
  GLuint floor_program_id = 0;
  CHECK_GL_ERROR(floor_program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(floor_program_id, vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(floor_program_id, floor_fragment_shader_id));
  CHECK_GL_ERROR(glAttachShader(floor_program_id, geometry_shader_id));

  // Bind attributes.
  CHECK_GL_ERROR(glBindAttribLocation(floor_program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindFragDataLocation(floor_program_id, 0, "fragment_color"));
  glLinkProgram(floor_program_id);
  CHECK_GL_PROGRAM_ERROR(floor_program_id);

  // Get the uniform locations.
  GLint floor_projection_matrix_location = 0;
  CHECK_GL_ERROR(floor_projection_matrix_location =
                     glGetUniformLocation(floor_program_id, "projection"));
  GLint floor_model_matrix_location = 0;
  CHECK_GL_ERROR(floor_model_matrix_location =
                     glGetUniformLocation(floor_program_id, "model"));
  GLint floor_view_matrix_location = 0;
  CHECK_GL_ERROR(floor_view_matrix_location =
                     glGetUniformLocation(floor_program_id, "view"));
  GLint floor_light_position_location = 0;
  CHECK_GL_ERROR(floor_light_position_location =
                     glGetUniformLocation(floor_program_id, "light_position"));

   // Let's create our ogre program.
  GLuint ogre_program_id = 0;
  CHECK_GL_ERROR(ogre_program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(ogre_program_id, vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(ogre_program_id, ogre_fragment_shader_id));
  CHECK_GL_ERROR(glAttachShader(ogre_program_id, geometry_shader_id));

  // Bind attributes.
  CHECK_GL_ERROR(glBindAttribLocation(ogre_program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindFragDataLocation(ogre_program_id, 0, "fragment_color"));
  glLinkProgram(ogre_program_id);
  CHECK_GL_PROGRAM_ERROR(ogre_program_id);

  // Get the uniform locations.
  GLint ogre_projection_matrix_location = 1;
  CHECK_GL_ERROR(ogre_projection_matrix_location =
                     glGetUniformLocation(ogre_program_id, "projection"));
  GLint ogre_model_matrix_location = 1;
  CHECK_GL_ERROR(ogre_model_matrix_location =
                     glGetUniformLocation(ogre_program_id, "model"));
  GLint ogre_view_matrix_location = 1;
  CHECK_GL_ERROR(ogre_view_matrix_location =
                     glGetUniformLocation(ogre_program_id, "view"));
  GLint ogre_light_position_location = 1;
  CHECK_GL_ERROR(ogre_light_position_location =
                     glGetUniformLocation(ogre_program_id, "light_position"));
  GLint ogre_com_location = 1;
  CHECK_GL_ERROR(ogre_com_location =
                     glGetUniformLocation(ogre_program_id, "center_of_mass"));
  GLint ogre_y_min_location = 1;
  CHECK_GL_ERROR(ogre_y_min_location =
                     glGetUniformLocation(ogre_program_id, "y_min"));
  GLint ogre_y_max_location = 1;
  CHECK_GL_ERROR(ogre_y_max_location =
                     glGetUniformLocation(ogre_program_id, "y_max"));
  ogre_texture_location = 1;
  CHECK_GL_ERROR(ogre_texture_location =
                     glGetUniformLocation(ogre_program_id, "texture_sampler"));
  ogre_texture_on_location = 1;
  CHECK_GL_ERROR(ogre_texture_on_location =
                     glGetUniformLocation(ogre_program_id, "is_texture_on"));

   // Let's create our bone program.
  GLuint bone_program_id = 0;
  CHECK_GL_ERROR(bone_program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(bone_program_id, vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(bone_program_id, bone_fragment_shader_id));
  CHECK_GL_ERROR(glAttachShader(bone_program_id, bone_geometry_shader_id));

  // Bind attributes.
  CHECK_GL_ERROR(glBindAttribLocation(bone_program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindFragDataLocation(bone_program_id, 0, "fragment_color"));
  glLinkProgram(bone_program_id);
  CHECK_GL_PROGRAM_ERROR(bone_program_id);

  // Get the uniform locations.
  GLint bone_projection_matrix_location = 2;
  CHECK_GL_ERROR(bone_projection_matrix_location =
                     glGetUniformLocation(bone_program_id, "projection"));
  GLint bone_model_matrix_location = 2;
  CHECK_GL_ERROR(bone_model_matrix_location =
                     glGetUniformLocation(bone_program_id, "model"));
  GLint bone_view_matrix_location = 2;
  CHECK_GL_ERROR(bone_view_matrix_location =
                     glGetUniformLocation(bone_program_id, "view"));
  glm::vec4 selected;
  GLint bone_selected_location = 2;
  CHECK_GL_ERROR(bone_selected_location =
                     glGetUniformLocation(bone_program_id, "selected"));

     // Let's create our cylinder program.
  GLuint cylinder_program_id = 0;
  CHECK_GL_ERROR(cylinder_program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(cylinder_program_id, vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(cylinder_program_id, cylinder_fragment_shader_id));
  CHECK_GL_ERROR(glAttachShader(cylinder_program_id, cylinder_geometry_shader_id));

  // Bind attributes.
  CHECK_GL_ERROR(glBindAttribLocation(cylinder_program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindFragDataLocation(cylinder_program_id, 0, "fragment_color"));
  glLinkProgram(cylinder_program_id);
  CHECK_GL_PROGRAM_ERROR(cylinder_program_id);

  // Get the uniform locations.
  GLint cylinder_projection_matrix_location = 3;
  CHECK_GL_ERROR(cylinder_projection_matrix_location =
                     glGetUniformLocation(cylinder_program_id, "projection"));
  GLint cylinder_model_matrix_location = 3;
  CHECK_GL_ERROR(cylinder_model_matrix_location =
                     glGetUniformLocation(cylinder_program_id, "model"));
  GLint cylinder_view_matrix_location = 3;
  CHECK_GL_ERROR(cylinder_view_matrix_location =
                     glGetUniformLocation(cylinder_program_id, "view"));
  glm::vec3 line_color;
  GLint color_location = 3;
  CHECK_GL_ERROR(color_location = glGetUniformLocation(cylinder_program_id, "line_color"));
 

  glm::vec4 light_position = glm::vec4(0.0f, 100.0f, 0.0f, 1.0f);

  center = glm::vec3(center_of_mass);

  while (!glfwWindowShouldClose(window)) {
    // Setup some basic window stuff.
    glfwGetFramebufferSize(window, &window_width, &window_height);
    glViewport(0, 0, window_width, window_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDepthFunc(GL_LESS);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glCullFace(GL_BACK);



    // Compute our view, and projection matrices.
    if (fps_mode)
      center = eye - camera_distance * look;
    else
      eye = center + camera_distance * look;
    selected = glm::vec4(0);
    if(!drag_state)
      selected_bone = Perform_Ray_Collision();
    if(selected_bone) {
      selected = selected_bone->first_world_point;
      std::cout << "\nMoused over bone " << selected_bone->id << " at " << selected << std::endl;
      std::cout << "Length of bone is " << selected_bone->length << std::endl;
    }

    view_matrix = glm::lookAt(eye, center, up);
    light_position = glm::vec4(eye, 1.0f);

    aspect = static_cast<float>(window_width) / window_height;
    projection_matrix =
        glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect, kNear, kFar);
    model_matrix = glm::mat4(1.0f);

    // Bind to our floor VAO.
    CHECK_GL_ERROR(glBindVertexArray(array_objects[kFloorVao]));

    // Use our program.
    CHECK_GL_ERROR(glUseProgram(floor_program_id));

    // Pass uniforms in.
    CHECK_GL_ERROR(glUniformMatrix4fv(floor_projection_matrix_location, 1,
                                      GL_FALSE, &projection_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(floor_model_matrix_location, 1, GL_FALSE,
                                      &floor_model_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(floor_view_matrix_location, 1, GL_FALSE,
                                      &view_matrix[0][0]));
    CHECK_GL_ERROR(
        glUniform4fv(floor_light_position_location, 1, &light_position[0]));

    // Draw our triangles.
    CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, floor_faces.size() * 3,
                                  GL_UNSIGNED_INT, 0));

    // eye_adjusted = glm::vec4(eye[0], eye[1], eye[2], 1);
    // bone_vertices[bone_vertices.size()-2] = eye_adjusted;
    // bone_vertices[bone_vertices.size()-1] = eye_adjusted + kFar*eye_ray;


    bone_vertices.clear();
    for(auto child : b_graph.root->children) {
      LoadBoneLines(child, bone_vertices, bone_lines, b_graph.root->translate,b_graph.root->translate * glm::vec4(0,0,0,1));
    }

    if(selected_bone) {
      glm::mat4 toBoneTransform = getDeformedTransform(selected_bone);
      for(int i = 0; i < cylinder_vertices.size(); ++i) {
        cylinder_vertices[i][1] *= selected_bone->length;
        cylinder_vertices[i] = toBoneTransform * cylinder_vertices[i];
      }
      std::vector<glm::vec4> axis_verts;
      std::vector<glm::uvec2> axis_indices;
      line_color = glm::vec3(0.0,1.0,1.0);
      glm::vec4 pt1 = cylinder_vertices.back();
      cylinder_vertices.pop_back();
      glm::vec4 pt2 = cylinder_vertices.back();
      cylinder_vertices.pop_back();
      glm::vec4 pt3 = cylinder_vertices.back();
      cylinder_vertices.pop_back();

      axis_verts.push_back(pt3); // center
      axis_verts.push_back(pt2);
      axis_indices.push_back(glm::uvec2(axis_verts.size()-2, axis_verts.size()-1));


      // Bind to our cylinder VAO.
      CHECK_GL_ERROR(glBindVertexArray(array_objects[kCylinderVao]));
      // Generate buffer objects
      CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kCylinderVao][0]));

      // Setup vertex data in a VBO.
      CHECK_GL_ERROR(
          glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kCylinderVao][kVertexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                  sizeof(float) * cylinder_vertices.size() * 4,
                                  &cylinder_vertices[0], GL_STATIC_DRAW));
      CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
      CHECK_GL_ERROR(glEnableVertexAttribArray(0));

      // Setup element array buffer.
      CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                  buffer_objects[kCylinderVao][kIndexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                  sizeof(uint32_t) * cylinder_faces.size() * 2,
                                  &cylinder_faces[0], GL_STATIC_DRAW));

      // Use our program.
      CHECK_GL_ERROR(glUseProgram(cylinder_program_id));

      // Pass uniforms in.
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_projection_matrix_location, 1,
                                        GL_FALSE, &projection_matrix[0][0]));
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_model_matrix_location, 1, GL_FALSE,
                                        &model_matrix[0][0]));
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_view_matrix_location, 1, GL_FALSE,
                                        &view_matrix[0][0]));
      CHECK_GL_ERROR(glUniform3fv(color_location, 1, &line_color[0]));
   
      // Draw our triangles.
      CHECK_GL_ERROR(glDrawElements(GL_LINES, cylinder_faces.size() * 2,
                                     GL_UNSIGNED_INT, 0));



      //DRAWING AXES
      line_color = glm::vec3(1.0, 0.0, 0.0);


      // Setup vertex data in a VBO.
      CHECK_GL_ERROR(
          glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kCylinderVao][kVertexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                  sizeof(float) * axis_verts.size() * 4,
                                  &axis_verts[0], GL_STATIC_DRAW));
      CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
      CHECK_GL_ERROR(glEnableVertexAttribArray(0));

      // Setup element array buffer.
      CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                  buffer_objects[kCylinderVao][kIndexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                  sizeof(uint32_t) * axis_indices.size() * 2,
                                  &axis_indices[0], GL_STATIC_DRAW));

      // Use our program.
      CHECK_GL_ERROR(glUseProgram(cylinder_program_id));

      // Pass uniforms in.
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_projection_matrix_location, 1,
                                        GL_FALSE, &projection_matrix[0][0]));
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_model_matrix_location, 1, GL_FALSE,
                                        &model_matrix[0][0]));
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_view_matrix_location, 1, GL_FALSE,
                                        &view_matrix[0][0]));
      CHECK_GL_ERROR(glUniform3fv(color_location, 1, &line_color[0]));

      // Draw our triangles.
      CHECK_GL_ERROR(glDrawElements(GL_LINES, axis_indices.size() * 2,
                                     GL_UNSIGNED_INT, 0));

      line_color = glm::vec3(0.0, 0.0, 1.0);

      axis_verts.clear();
      axis_indices.clear();
      axis_verts.push_back(pt3); // center
      axis_verts.push_back(pt1);
      axis_indices.push_back(glm::uvec2(axis_verts.size()-2, axis_verts.size()-1));

      // Setup vertex data in a VBO.
      CHECK_GL_ERROR(
          glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kCylinderVao][kVertexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                  sizeof(float) * axis_verts.size() * 4,
                                  &axis_verts[0], GL_STATIC_DRAW));
      CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
      CHECK_GL_ERROR(glEnableVertexAttribArray(0));

      // Setup element array buffer.
      CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                  buffer_objects[kCylinderVao][kIndexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                  sizeof(uint32_t) * axis_indices.size() * 2,
                                  &axis_indices[0], GL_STATIC_DRAW));

      // Use our program.
      CHECK_GL_ERROR(glUseProgram(cylinder_program_id));

      // Pass uniforms in.
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_projection_matrix_location, 1,
                                        GL_FALSE, &projection_matrix[0][0]));
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_model_matrix_location, 1, GL_FALSE,
                                        &model_matrix[0][0]));
      CHECK_GL_ERROR(glUniformMatrix4fv(cylinder_view_matrix_location, 1, GL_FALSE,
                                        &view_matrix[0][0]));
      CHECK_GL_ERROR(glUniform3fv(color_location, 1, &line_color[0]));

      // Draw our triangles.
      CHECK_GL_ERROR(glDrawElements(GL_LINES, axis_indices.size() * 2,
                                     GL_UNSIGNED_INT, 0));



      cylinder_vertices.push_back(pt3);
      cylinder_vertices.push_back(pt2);
      cylinder_vertices.push_back(pt1);
    }

    cylinder_vertices = origin_cylinder_vertices;


    // Bind to our bone VAO.
    CHECK_GL_ERROR(glBindVertexArray(array_objects[kBoneVao]));

    // Use our program.
    CHECK_GL_ERROR(glUseProgram(bone_program_id));

    // Pass uniforms in.
    CHECK_GL_ERROR(glUniformMatrix4fv(bone_projection_matrix_location, 1,
                                      GL_FALSE, &projection_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(bone_model_matrix_location, 1, GL_FALSE,
                                      &model_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(bone_view_matrix_location, 1, GL_FALSE,
                                      &view_matrix[0][0]));
    CHECK_GL_ERROR(glUniform4fv(bone_selected_location, 1,
                                      &selected[0]));


    // Generate buffer objects
    CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kBoneVao][0]));

    // Setup vertex data in a VBO.
    CHECK_GL_ERROR(
        glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kBoneVao][kVertexBuffer]));
    CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                sizeof(float) * bone_vertices.size() * 4,
                                &bone_vertices[0], GL_STATIC_DRAW));
    CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
    CHECK_GL_ERROR(glEnableVertexAttribArray(0));

    // Setup element array buffer.
    CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                buffer_objects[kBoneVao][kIndexBuffer]));
    CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                sizeof(uint32_t) * bone_lines.size() * 2,
                                &bone_lines[0], GL_STATIC_DRAW));

    // Draw our lines.
    CHECK_GL_ERROR(glDrawElements(GL_LINES, bone_lines.size() * 2,
                                  GL_UNSIGNED_INT, 0));

    RunLinearSkinning(ogre_vertices);
    
    // Bind to our ogre VAO.
    CHECK_GL_ERROR(glBindVertexArray(array_objects[kOgreVao]));

    // Use our program.
    CHECK_GL_ERROR(glUseProgram(ogre_program_id));

    // Pass uniforms in.
    CHECK_GL_ERROR(glUniformMatrix4fv(ogre_projection_matrix_location, 1,
                                      GL_FALSE, &projection_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(ogre_model_matrix_location, 1, GL_FALSE,
                                      &model_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(ogre_view_matrix_location, 1, GL_FALSE,
                                      &view_matrix[0][0]));
    CHECK_GL_ERROR(glUniform4fv(ogre_light_position_location, 1, &light_position[0]));

    CHECK_GL_ERROR(glUniform4fv(ogre_com_location, 1, &center_of_mass[0]));

    CHECK_GL_ERROR(glUniform1f(ogre_y_max_location, y_max));

    CHECK_GL_ERROR(glUniform1f(ogre_y_min_location, y_min));

    // Generate buffer objects
    CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kOgreVao][0]));

    // Setup vertex data in a VBO.
    CHECK_GL_ERROR(
        glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kOgreVao][kVertexBuffer]));
    CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                sizeof(float) * ogre_vertices.size() * 4,
                                &ogre_vertices[0], GL_STATIC_DRAW));
    CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
    CHECK_GL_ERROR(glEnableVertexAttribArray(0));

    // Setup element array buffer.
    CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                buffer_objects[kOgreVao][kIndexBuffer]));
    CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                sizeof(uint32_t) * ogre_faces.size() * 3,
                                &ogre_faces[0], GL_STATIC_DRAW));

    // Draw our triangles.
    CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, ogre_faces.size() * 3,
                                  GL_UNSIGNED_INT, 0));
    // Poll and swap.
    glfwPollEvents();
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  for (int i = 0; i < images.size(); ++i) delete[] images[i].bytes;
  exit(EXIT_SUCCESS);
}
