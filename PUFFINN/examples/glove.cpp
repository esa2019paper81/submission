// An example querying for a random vector in a dataset.

#include <fstream>
#include <iostream>
#include <vector>
#include "collection.hpp"
#include "similarity_measure/cosine.hpp"

// Read a vector collection in the format used by GloVe.
std::vector<puffinn::UnitVector> read_glove_data(const std::string& filename, int dimensions) {
    std::ifstream file(filename);

    std::vector<puffinn::UnitVector> res;
    if (!file.is_open()) {
        printf("File %s not found\n", filename.c_str());
        return res;
    }

    std::string id;
    file >> id;
    while (!file.eof()) {
        std::vector<float> row;
        float tmp;
        for (int i=0; i<dimensions; i++) {
            file >> tmp;
            row.push_back(tmp);
        }
        res.push_back(puffinn::UnitVector(row));
        file >> id;
    }
    return res;
}

int main() {
    const std::string FILENAME = "glove.6B.100d.txt";
    const int DIMENSIONS = 100;
    const unsigned long long GB = 1024*1024*1024;
    const float RECALL = 0.5;

    auto vectors = read_glove_data(FILENAME, DIMENSIONS);
    puffinn::LSHTable<puffinn::CosineSimilarity> lsh(
        DIMENSIONS,
        4*GB
    );
    for (auto v : vectors) { lsh.insert(v); }
    lsh.rebuild();

    auto query = puffinn::UnitVector::generate_random(DIMENSIONS);
    auto res = lsh.search_k(query, 10, RECALL);
    std::cout << "Found the following near neighbors:";
    for (auto idx : res) {
        std::cout << " " << idx;
    }
    std::cout << std::endl;

}
