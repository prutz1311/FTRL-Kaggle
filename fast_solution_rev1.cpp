/* Run with: ./fast_solution_rev1 train test submission
 * 
 * To set parameters, edit params.cfg
 * 
 * If the number of columns in test file differs from 23,
 * change #define ARRAY_SIZE to that value.
 * */

#include <iostream>
#include <vector>
#include <cmath>
#include <stdbool.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <functional>
#include <array>
#include <map>
#include <ctime>
#include <iomanip>
#include <stdio.h>


#define ARRAY_SIZE 23

class next_line_data {
public:
    long long                               t;
    std::string                             ID;
    std::array<unsigned long, ARRAY_SIZE>   x;
    double                                  y; 
    next_line_data (unsigned long t_arg,
                    std::string ID_arg,
                    std::array<unsigned long,ARRAY_SIZE>& x_arg,
                    double y_arg):
                    t(t_arg), ID(ID_arg), x(x_arg), y(y_arg) {}
    next_line_data() {}
};

class ftrl_proximal {
public:
    double  alpha;
    double  beta;
    double  L1;
    double  L2;
    int     D;
    bool    interaction;
    double* z;
    double* n;
    double* w;
    std::vector<unsigned long> indices;
    
    ftrl_proximal(double alpha, double beta, double L1, double L2, int D, bool interaction=false) {
        this->alpha         = alpha;
        this->beta          = beta;
        this->L1            = L1;
        this->L2            = L2;
        this->D             = D;
        this->interaction   = interaction;
        this->n             = (double*)calloc(D, sizeof(double));
        this->z             = (double*)calloc(D, sizeof(double));
        this->w             = (double*)calloc(D, sizeof(double));
    }
    
    void update_indices(std::array<unsigned long, ARRAY_SIZE>& x) {
        unsigned int L = ARRAY_SIZE;
        indices.clear();
        for (unsigned int c=0; c<ARRAY_SIZE; c++) {
            indices.push_back(x[c]);
        }
        
        if (this->interaction) {
            for (unsigned int i=1; i<L; i++) {
                for (unsigned int j=i+1; j<L; j++) {
                    indices.push_back((i * j) % D);
                }
            }
        }
        
    }
    
    double predict(std::array<unsigned long, ARRAY_SIZE>& x) {
        double wTx  = 0;
        int sign;
        unsigned long i;
        
        this->update_indices(x);
        for (unsigned int c=0; c<indices.size(); c++) {
            i = indices[c];
            if (z[i] < 0) sign = -1.0; else sign = 1.0;
            
            if (sign * z[i] <= L1) w[i] = 0.0;
            else w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2);
            wTx += w[i];
        }
        return 1. / (1. + exp(-std::max(std::min(wTx, 35.), -35.)));
    }
    
    void update(std::array<unsigned long, ARRAY_SIZE>& x, double p, double y) {
        double g = p - y;
        double sigma;
        int i;
        
        this->update_indices(x);
        for (unsigned int c=0; c<indices.size(); c++) {
            i = indices[c];
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha;
            z[i] += g - sigma * w[i];
            n[i] += g * g;
        }
    }
    
    ~ftrl_proximal() {
        free(n);
        free(z);
        free(w);
    }
};

double logloss(double p, double y) {
    p = std::max(std::min(p, 1. - 10e-15), 10e-15);
    if (y == 1.0) return -log(p); else return -log(1.0 - p);
}

long string_hash(std::string a) {
    register long len;
    register unsigned char *p;
    register long x;

    len = a.size();
    p = (unsigned char *) a.c_str();
    x = *p << 7;
    while (--len >= 0)
        x = (1000003*x) ^ *p++;
    x ^= a.size();
    if (x == -1)
        x = -2;
    return x;
}

std::vector<std::string> splitter(std::string & s) {
    std::vector<std::string> tokens;
    const char * str = s.c_str();
    char * pch;
    if (s.size() >= 20) {
        if (s.substr(20) == "10017406487169876828") {
            std::cout << s << std::endl;
            exit(0);
        }
    }
    pch = strtok((char*)str,",");
    while (pch != NULL)
    {
        tokens.push_back(std::string(pch));
        pch = strtok (NULL, ",");
    }
    return tokens;
}


next_line_data data (std::string path, unsigned long D, bool train=true) { 
    static unsigned long t;
    std::string ID, s;
    std::array<unsigned long, ARRAY_SIZE> x = {0};
    double y = 0.0;
    static std::ifstream f;
    unsigned int i = 0;
    static std::vector<std::string> header;
    std::tuple<unsigned long, std::string, std::array<unsigned long, ARRAY_SIZE>, double> to_return;
    std::vector<std::string>   result;
    std::string                line;
    std::string                cell;
    if (!f.is_open()) {
        f.open(path.c_str());
        std::getline(f, s);
        if (s.size() != 0) {
            while (s[s.size()-1] == '\n' || s[s.size()-1] == '\r') {
                s.erase(s.end()-1);
            }
        }
        header = splitter(s);
    }
    
    if (f.eof()) {
        f.close();
        t = 0;
        return next_line_data(t-1, std::string(), x, y);
    }
    
    std::getline(f, s);
    if (s.size() != 0) {
        while (s[s.size()-1] == '\n' || s[s.size()-1] == '\r') {
            s.erase(s.end()-1);
        }
    }
    result = splitter(s);
    
    if (s.size() < 2) {
        f.close();
        t = 0;
        return next_line_data(t-1, std::string(), x, y);
    }
    
    ID = result[0];
    if (train) {
        if (result[1] == "1") y = 1.0;
        
        result[2] = result[2].substr(6);
        x[0] = 0;
        for (i=2; i<ARRAY_SIZE+1; i++) {
            x[i-1] = std::abs(string_hash(header[i] + std::string("_") + result[i])) % D;
        }
    }
    else {
        result[1] = result[1].substr(6);
        x[0] = 0;
        for (i=1; i<ARRAY_SIZE; i++) {
            x[i] = std::abs(string_hash(header[i] + std::string("_") + result[i])) % D;
        }
    }
    t++;
    return next_line_data(t-1, ID, x, y);
}
    

int main(int argc, char **argv)
{
    std::map<std::string, std::string> params;
    const char * train      = argv[1];
    const char * test       = argv[2];
    const char * submission = argv[3];
    std::ifstream config("params.cfg");
    std::string key, value;
    next_line_data row;
    double alpha, beta, L1, L2;
    unsigned long D, epoch, holdout;
    bool do_interactions;
    double p;
    
    while (!config.eof()) {
        std::getline(config,key,'=');
        std::getline(config,value,'\n');
        params[key] = value;
    }
    std::stringstream(params[std::string("alpha")]) >> alpha;
    std::stringstream(params[std::string("beta")]) >> beta;
    std::stringstream(params[std::string("L1")]) >> L1;
    std::stringstream(params[std::string("L2")]) >> L2;
    std::stringstream(params[std::string("D")]) >> D;
    std::stringstream(params[std::string("epoch")]) >> epoch;
    std::stringstream(params[std::string("holdout")]) >> holdout;
    do_interactions = (params[std::string("do_interactions")] == "true");
    time_t start = time(0), end, now;
    
    ftrl_proximal learner = ftrl_proximal(alpha, beta, L1, L2, D, do_interactions);
    for (unsigned long e=0; e<epoch; e++) {
        double loss = 0.0;
        unsigned long count = 0;
        do {
            row = data(train, D);  // проверить генерацию row.x при row.t == 200 и 300 и, может быть, везде
            p = learner.predict(row.x);
            if (row.t % holdout == 0) {
                loss += logloss(p, row.y);
                count++;
            }
            else {
                if (row.t != -1) {
                    learner.update(row.x, p, row.y);
                }
            }
            if (row.t % 250000 == 0 && row.t > 1) {
                now = time(0);
                printf(" %s\tencountered: %lld\tcurrent logloss: %f\n" , std::asctime(std::localtime(&now)), row.t, loss/count);
            }
        } while (row.t !=-1 && row.ID != "");
        end = time(0);
        printf("Epoch %lu finished, holdout logloss: %f, elapsed time: %f\n", e, loss/count, difftime(end, start));
    }
    
    std::cout << "Predicting...\n";
    std::ofstream outfile(submission);
    
    outfile << "id,click\n";
    do {
        row = data(test, D, false);
        if (row.t == -1) {
            break;
        }
        p = learner.predict(row.x);
        outfile << row.ID << "," << std::fixed << std::setprecision(13) << p <<std::endl;
    } while (true);
    return 0;
}
