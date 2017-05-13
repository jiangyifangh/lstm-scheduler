#include <random>
#include <iostream>
#include <cmath>
#include <vector>
#include <math.h>
#include <string>
#include <set>


#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>

using namespace std;
// define the error information

#define SOURCE_NODE_ID 0
#define SINK_NODE_ID 1

#define NO_DATA -1
#define INPUT 0
#define PREV_OUTPUT 1
#define OUTPUT 2
#define PREV_STATE 3
#define STATE 4
#define WEIGHT_DATA 5
#define BIAS_DATA 6

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

float one = 1.0;
float zero = 0.0;

// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
    return 1.f / (1.f + expf(-in));
}

__forceinline__ __device__ float mulf(float a, float b) {
    return a * b;
}

__forceinline__ __device__ float addf(float a, float b) {
    return a + b;
}

class Node;
class NodeDef;
class FuseDef;
class Edge;
class LSTMNetwork;
class LSTMCell;

class NodeDef {
public:
    string name;
    string op;
    int data_type;

    void set_name(string name) {
        this->name = name;
    }

    void set_op(string op) {
        this->op = op;
    }

    void set_data_type(int data_type) {
        this->data_type = data_type;
    }
};

class Node {
public:
    string name;
    string op;
    int data_type;

    float* tmp;

    // TODO: whether the remaining nodes are element-wise
    bool element_left;
    bool is_data;

    vector<Edge*> in_edges;
    vector<Edge*> out_edges;

    Node(NodeDef def);
    bool isMatmulOp();

//    Status Node::input_edge(int idx, const Edge** e) const {
//        for (auto edge : in_edges()) {
//            if (edge->dst_input() == idx) {
//                *e = edge;
//                return Status::OK();
//            }
//        }
//        return errors::NotFound("not found input edge ", idx);
//    }
//
//    Status Node::input_node(int idx, const Node** n) const {
//        const Edge* e;
//        input_edge(idx, &e);
//        if (e == nullptr) {
//            *n = nullptr;
//        } else {
//            *n = e->src();
//        }
//        return Status::OK();
//    }
    bool isActivated();
};

class Edge {
public:
    Node *src;
    Node *dst;
    bool activated = false;
    bool status = false;
    int edge_id = -1;

    bool isActivated() {
        return status;
    }

    void reset(){
        status = false;
    }

    void activate() {
        status = true;
    }

    void set_activation(bool b) {
        activated = b;
    }
};

Node::Node(NodeDef def) {
    this->name = def.name;
    this->op = def.op;
    this->data_type = def.data_type;
}

bool Node::isMatmulOp() {
    return (this->op == "matmul");
}

bool Node::isActivated() {
    for (auto edge : this->in_edges) {
        if (!edge->isActivated())
            return false;
    }

    return true;
}

class LSTMNetwork{
public:
    cublasHandle_t handle;

    // arguments about the network
    int num_layers;
    int mem_cell_num;
    int input_dim;
    int mini_batch;
    int seq_length;
    int num_elements;

    float *input_data;  // required. (seqLength) * (numLayers + 1) * numElements
    float *output_data; // required. (seqLength + 1) * (numLayers) * numElements
    float *cell_state;  // optional. (seqLength + 1) * (numLayers) * numElements

    // weight dimension: num_layers * mem_cell_num * input_dim

    // workspace for the result of R * h and W * x
    float *tmp_h;
    float *temp_input;

    vector<vector<LSTMCell*>> lstmCells;

    // W and R
    // W is for the input
    // R is for the h_prev
    // TODO: is it necessary to have another work space for weight
    float *weight_in;
    float *weight_out;

    float *bias;


    LSTMNetwork(int seq_length, int num_layers, int mem_cell_num, int input_dim, int mini_batch);
    ~LSTMNetwork();

    void feedforward();
    void backprop();

};

class LSTMCell {
public:
    LSTMCell();

    cublasHandle_t handle;

    // arguments about the network
    int num_layers;
    int mem_cell_num;
    int input_dim;
    int mini_batch;
    int seq_length;
    int num_elements;

    //device memory
    float *input_data;  // required. (seqLength) * (numLayers + 1) * numElements
    float *output_data; // required. (seqLength + 1) * (numLayers) * numElements
    float *cell_state;  // optional. (seqLength + 1) * (numLayers) * numElements

    // weight dimension: num_layers * mem_cell_num * input_dim

    // workspace for the result of R * h and W * x
    float *tmp_h;
    float *temp_input;

    int seq_num;
    int layer_num;

    int num_matmul; // Number of matrix multiplication operator in LSTM Cell

    // TODO: not necessary needed
    vector<Node*> nodes;
    vector<Edge*> edges;

    Node *src;
    Node *sink;

    LSTMCell(int seq_num, int layer_num);

    Node* addEndpoint(string name, int id) {
        NodeDef def;
        def.set_name(name);
        def.set_op("NoOp");
        def.set_data_type(NO_DATA);

        Node* node = addNode(def);
        return node;
    }

    Node *addDataPoint(string name, int data_type) {
        NodeDef def;
        def.set_name(name);
        def.set_op("NoOp");
        def.set_data_type(data_type);

        Node* node = addNode(def);
        return node;
    }

    static Node *addOp(string name, string op_type) {
        NodeDef def;
        def.set_name(name);
        def.set_op(op_type);
        def.set_data_type(NO_DATA);

        Node* node = addNode(def);
        return node;
    }

    static Node *addNode(NodeDef def) {
        Node *node = new Node(def);
        return node;
    };

    Node *sigmoid(Node *node);

    Node *elPlus(Node *a, Node *b);

    Node *matmul(Node *a, Node *b);

    void transfer(Node *out_node, Node *in_node);

    void binary_transfer(Node *in_node_a, Node *in_node_b, Node *out_node);

    Node *tanh(Node *in_node);

    Node *elMul(Node *a, Node *b);

    void outputData(Node *pre_data, Node *data);

    void connectSource(Node *node);

    void connectSink(Node *node);

    void preprocess();

    void printGraph();
};

LSTMCell::LSTMCell(int seq_num, int layer_num) {
    this->seq_num = seq_num;
    this->layer_num = layer_num;

    src = addEndpoint("_SOURCE", SOURCE_NODE_ID);
    sink = addEndpoint("_SINK", SINK_NODE_ID);

    // TODO: connect source and sink to DAG
}

// Constructor for cell definition
LSTMCell::LSTMCell() {
    src = addEndpoint("_SOURCE", SOURCE_NODE_ID);
    sink = addEndpoint("_SINK", SINK_NODE_ID);
}

void LSTMCell::transfer(Node *in_node, Node *out_node) {
    Edge *e = new Edge();
    e->src = in_node;
    e->dst = out_node;
    in_node->out_edges.push_back(e);
    out_node->in_edges.push_back(e);
}

void LSTMCell::binary_transfer(Node *in_node_a, Node *in_node_b, Node *out_node) {
    Edge *e_a = new Edge();
    Edge *e_b = new Edge();

    e_a->src = in_node_a;
    e_a->dst = out_node;
    e_b->src = in_node_b;
    e_b->dst = out_node;
    if (e_a->src->data_type != NO_DATA){
        e_a->set_activation(true);
    }

    if (e_b->src->data_type != NO_DATA){
        e_b->set_activation(true);
    }

    in_node_a->out_edges.push_back(e_a);
    in_node_b->out_edges.push_back(e_b);
    out_node->in_edges.push_back(e_a);
    out_node->in_edges.push_back(e_b);
}

Node *LSTMCell::sigmoid(Node *in_node) {
    Node *node = LSTMCell::addOp("sigmoid", "sigmoid");
    LSTMCell::transfer(in_node, node);
    return node;
}

Node *LSTMCell::tanh(Node *in_node) {
    Node *node = LSTMCell::addOp("tanh", "tanh");
    LSTMCell::transfer(in_node, node);
    return node;
}

Node *LSTMCell::elPlus(Node *a, Node *b) {
    Node *node = LSTMCell::addOp("element-wise plus", "plus");
    LSTMCell::binary_transfer(a, b, node);
    return node;
}

Node *LSTMCell::matmul(Node *a, Node *b) {
    Node *node = LSTMCell::addOp("matrix multiplication", "matmul");
    LSTMCell::binary_transfer(a, b, node);
    return node;
}

Node *LSTMCell::elMul(Node *a, Node *b) {
    Node *node = LSTMCell::addOp("element-wise multiplication", "mul");
    LSTMCell::binary_transfer(a, b, node);
    return node;
}

void LSTMCell::outputData(Node *pre_data, Node *data) {
    LSTMCell::transfer(pre_data, data);
}

void LSTMCell::connectSource(Node *node) {
    Edge *e = new Edge();
    e->src = this->src;
    e->dst = node;
    this->src->out_edges.push_back(e);
    node->in_edges.push_back(e);
}

void LSTMCell::connectSink(Node *node) {
    Edge *e = new Edge();
    e->src = node;
    e->dst = this->sink;
    node->out_edges.push_back(e);
    this->sink->in_edges.push_back(e);
}

// TODO: Only support classic LSTM currently
void LSTMCell::preprocess() {
    // TODO: Fuse node definition
//    FuseDef fuseDef;
    NodeDef def;
    def.data_type = NO_DATA;
    def.name = "Fused element-wise";
    def.op = "fuse";
    Node *fused_node = new Node(def);

    // Arrive all the approachable matrix multiplication operators
    set<Node*> *barrier = new set<Node*>;
    set<Node*> *unactivated = new set<Node*>;
    set<Node*> *cur_frontier = new set<Node*>;
    set<Node*> *next_frontier = new set<Node*>;

    for (auto edge : src->out_edges) {
        cur_frontier->insert(edge->dst);
    }

    // TODO: attribute of LSTMCELL
    int tmp_mem_count = 0;

    while (!cur_frontier->empty()) {
        for (auto node : *cur_frontier) {
            for (auto edge : node->out_edges) {
                edge->activate();
                if (edge->dst->isMatmulOp()) {
                    barrier->insert(edge->dst);
                }
            }
        }

        // switch cur_frontier and next_frontier
        set<Node*> *tmp = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp;
        next_frontier->clear();
    }

    for (auto node : *barrier)
        cout << node->op << endl;

    // Barrier is not empty
    tmp_mem_count += barrier->size();




    // traverse the remaining graph to build fuse node
    // connect inputs to fused node, connect fused node to outputs
    

}

// print graph using BFS
void LSTMCell::printGraph() {
    set<Node*> *cur_frontier = new set<Node*>;
    set<Node*> *next_frontier = new set<Node*>;

    int count = 1;
    cur_frontier->insert(src);
    while (!cur_frontier->empty()) {
        printf("Frontier %d:\n", count);

        for (auto node : *cur_frontier) {
            for (auto edge : node->out_edges) {
                next_frontier->insert(edge->dst);
            }

            cout << "Node: " << node->name << endl;
        }

        set<Node*> *tmp = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp;
        next_frontier->clear();

        count++;
    }
}

LSTMNetwork::LSTMNetwork(int seq_length, int num_layers, int mem_cell_num, int input_dim, int mini_batch) {
    this->num_layers = num_layers;
    this->mem_cell_num = mem_cell_num;
    this->input_dim = input_dim;
    this->mini_batch = mini_batch;
    this->seq_length = seq_length;

    // initialize the handle
    cublasErrCheck(cublasCreate(&this->handle));

    this->num_elements = input_dim * mini_batch;

    cudaErrCheck(cudaMalloc((void**)&output_data, (seq_length + 1) * (num_layers) * num_elements * sizeof(float)));

    cudaErrCheck(cudaMalloc((void**)&input_data, (seq_length) * (num_layers + 1) * num_elements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&cell_state, (seq_length + 1) * (num_layers) * num_elements * sizeof(float)));

//	cudaErrCheck(cudaMalloc((void**)&tmp_h, 4 * num_layers * num_elements * sizeof(float)));
//	cudaErrCheck(cudaMalloc((void**)&temp_input, 4 * seq_length * num_elements * sizeof(float)));

//	cudaErrCheck(cudaMalloc((void**)&bias, num_layers * mem_cell_num * 8 * sizeof(float)));

    // randomlize the weight
//	curandGenerator_t rng;
//	curandErrCheck(curandGenerateUniform(rng, this->weight, num_layers * mem_cell_num * input_dim * 8));
}

LSTMNetwork::~LSTMNetwork() {
    cublasErrCheck(cublasDestroy(this->handle));
    cudaErrCheck(cudaFree(output_data));
    cudaErrCheck(cudaFree(input_data));
    cudaErrCheck(cudaFree(cell_state));

//    cudaErrCheck(cudaFree(weight));
//    cudaErrCheck(cudaFree(bias));
}

int main(int argc, char* argv[]) {
    // arguments about the network
    int num_layers;
    int mem_cell_num;
    int input_dim;
    int mini_batch;
    int seq_length;

    if (argc == 6) {
        seq_length = atoi(argv[1]);
        num_layers =  atoi(argv[2]);
        mem_cell_num =  atoi(argv[3]);
        input_dim = atoi(argv[4]);
        mini_batch =  atoi(argv[5]);
    }
    else if (argc == 1) {
        printf("Running with default settings\n");
        seq_length = 100;
        num_layers =  4;
        mem_cell_num =  512;
        input_dim = 512;
        mini_batch =  64;
    }
    else {
        printf("Usage: 5 <seqLength> <numLayers> <hiddenSize> <inputSize> <miniBatch>\n");
        return 1;
    }

    printf("seqLength %d, numLayers %d, hidden_size %d, inputSize %d, miniBatch %d\n", seq_length, num_layers, mem_cell_num, input_dim, mini_batch);

    LSTMNetwork network(seq_length, num_layers, mem_cell_num, input_dim, mini_batch);

    // Configure LSTM cell
    LSTMCell lstm_cell = LSTMCell::LSTMCell();

    // Set number of weights
    lstm_cell.num_matmul = 8;

    // Define LSTM cell
    // TODO: type = weight data?

    Node *h_prev = lstm_cell.addDataPoint("h_prev", PREV_OUTPUT);
    Node *h = lstm_cell.addDataPoint("h", OUTPUT);
    Node *input = lstm_cell.addDataPoint("input", INPUT);
    Node *state_prev = lstm_cell.addDataPoint("state_prev", PREV_STATE);
    Node *state = lstm_cell.addDataPoint("state", STATE);

    Node *w_f = lstm_cell.addDataPoint("w_f", WEIGHT_DATA);
    Node *w_i = lstm_cell.addDataPoint("w_i", WEIGHT_DATA);
    Node *w_c = lstm_cell.addDataPoint("w_c", WEIGHT_DATA);
    Node *w_o = lstm_cell.addDataPoint("w_o", WEIGHT_DATA);
    Node *r_f = lstm_cell.addDataPoint("r_f", WEIGHT_DATA);
    Node *r_i = lstm_cell.addDataPoint("r_i", WEIGHT_DATA);
    Node *r_c = lstm_cell.addDataPoint("r_c", WEIGHT_DATA);
    Node *r_o = lstm_cell.addDataPoint("r_o", WEIGHT_DATA);

    Node *b_f = lstm_cell.addDataPoint("b_f", BIAS_DATA);
    Node *b_i = lstm_cell.addDataPoint("b_i", BIAS_DATA);
    Node *b_c = lstm_cell.addDataPoint("b_c", BIAS_DATA);
    Node *b_o = lstm_cell.addDataPoint("b_o", BIAS_DATA);

    Node *f = lstm_cell.sigmoid(
            lstm_cell.elPlus(
                    lstm_cell.elPlus(
                            lstm_cell.matmul(w_f, input),
                            lstm_cell.matmul(r_f, h_prev)),
                    b_f)
    );

    Node *i = lstm_cell.sigmoid(
            lstm_cell.elPlus(
                    lstm_cell.elPlus(
                            lstm_cell.matmul(w_i, input),
                            lstm_cell.matmul(r_i, h_prev)),
                    b_i)
    );

    Node *c = lstm_cell.tanh(
            lstm_cell.elPlus(
                    lstm_cell.elPlus(
                            lstm_cell.matmul(w_c, input),
                            lstm_cell.matmul(r_c, h_prev)),
                    b_c)
    );

    Node *o = lstm_cell.sigmoid(
            lstm_cell.elPlus(
                    lstm_cell.elPlus(
                            lstm_cell.matmul(w_o, input),
                            lstm_cell.matmul(r_o, h_prev)),
                    b_o)
    );

    // TODO: set this node's type to internal state
    Node *pre_state = lstm_cell.elPlus(
                    lstm_cell.elMul(f, state_prev),
                    lstm_cell.elMul(i, c)
    );

    // TODO: set this node's type to output
    Node *pre_h = lstm_cell.elMul(
                o,
                lstm_cell.tanh(pre_state)
    );

    lstm_cell.outputData(pre_state, state);
    lstm_cell.outputData(pre_h, h);

    lstm_cell.connectSource(h_prev);
    lstm_cell.connectSource(state_prev);
    lstm_cell.connectSource(input);
    lstm_cell.connectSink(h);
    lstm_cell.connectSink(state);

    lstm_cell.preprocess();

    //TODO: elementwise的decode和encode

    

    return 0;
}



