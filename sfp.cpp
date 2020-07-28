#include <iostream>
#include <complex>
#include <vector>
#include <map>
#include <Windows.h>

#define EIGEN_VECTORIZE_AVX
#define PI 3.141592653589793238

#include "Eigen/Eigen"


/*

     x  x+1

y    1   2
    
y+1  4   8

edge
    
     x        x+1

y    +----0----+
     |         |
     3         1
     |         |
y+1  +----2----+

*/

const int connecttable[16][4] = {
    {-1,-1,-1,-1},
    {-1,0,3,-1},
    {-1,-1,1,0},
    {-1,1,-1,3},
    {3,2,-1,-1},//4
    {0,-1,2,-1},
    {1,0,3,2},//c>0
    {1,-1,-1,2},
    {1,-1,-1,2},//8
    {3,2,1,0},//c>0
    {0,-1,2,-1},
    {3,2,-1,-1},
    {-1,1,-1,3},//12
    {-1,-1,1,0},
    {-1,0,3,-1},
    {-1,-1,-1,-1},
};

const int dirx[4] = { 0,1,0,-1 };
const int diry[4] = { -1,0,1,0 };

const int solveit = 5;
constexpr int solvediv = 2 << 5;

struct int2 {
    int x, y;
};

bool operator <(const int2& a, const int2& b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

struct surfacepoint_t {
    int i;
    int stateindex;
    int nx, ny;
    int dir;
    double kx, ky;
};

struct state_t {
    int i;
    int nx, ny;
    double kx, ky;
    double length;
    double grad;
    Eigen::VectorXcd eigenvector;
};

struct json_t {
    int type;
    std::map<std::string, json_t> obj; //1
    std::vector<json_t> vec; //2 
    std::string context; //0
};

struct hopping_t {
    int from, to, tx, ty;
    std::complex<double> value;
};

struct tb_t {
    int atomn, kp;
    std::map<std::string, int> atoms;
    std::vector<hopping_t> hoppings;
};

struct sfp_t {
    int gridx, gridy;
    int orbitalcount, atomcount;
    int thread;
    double chemp, invtemp, width;
    double u, u_, j, j_;
    tb_t tb;
    Eigen::VectorXd* eigenvalue, * eigenvaluecenter;
    Eigen::MatrixXcd* eigenvector;
    Eigen::MatrixXcd uc, us;
    Eigen::MatrixXd gapmat;
    Eigen::VectorXd gapfunc;
    std::vector<double> qpweight;
    std::vector<std::vector<surfacepoint_t>> surface;
    std::vector<state_t> states;
    std::map<int2, int> pairindex;
    std::vector<int2> pair;
    std::vector<Eigen::MatrixXcd> sus;
};

struct susmission_t {
    sfp_t* sfp;
    int s, e;
};

int cmp(const char* s1, const char* s2, int nof2) {
    for (int i = 0; i < nof2; i++) {
        if (!s1[i] || s1[i] != s2[i]) {
            return 1;
        }
    }
    return 0;
}

int find(const char* text, int textlength, const char* tag, int taglength) {
    int i;
    for (i = 0; i <= textlength - taglength; i++) {
        if (!cmp(text + i, tag, taglength)) {
            return i;
        }
    }
    return -1;
}

int read(const char* str, int length, json_t& json) {
    int pos, cur;
    if (length > 0) {
        if (str[0] == '{') {
            json.type = 1;
            cur = 1;
            while (str[cur] != '}') {
                std::string key;
                json_t temp;
                int next;
                pos = find(str + cur, length - cur, ":", 1);
                if (pos > 0) {
                    if (pos > 1 && str[cur + pos - 1] == '"' && str[cur] == '"') {
                        key = std::string(str + cur + 1, pos - 2);
                    }
                    else {
                        key = std::string(str + cur, pos);
                    }
                    cur += pos + 1;
                }
                else {
                    return -1;
                }
                next = read(str + cur, length - cur, temp);
                if (next >= 0) {
                    cur += next;
                }
                else {
                    return -1;
                }
                json.obj.insert(std::make_pair(key, temp));
                if (str[cur] == ',') {
                    cur++;
                }
            }
            return cur + 1;
        }
        else if (str[0] == '[') {
            json.type = 2;
            cur = 1;
            while (str[cur] != ']') {
                json_t temp;
                int next;
                next = read(str + cur, length - cur, temp);
                if (next >= 0) {
                    cur += next;
                }
                else {
                    return -1;
                }
                json.vec.push_back(temp);
                if (str[cur] == ',') {
                    cur++;
                }
            }
            return cur + 1;
        }
        else {
            int i;
            int instring = 0;
            //if (length > 0 && str[0] == '\"') {
            //	instring = 1;
            //}
            for (i = 0; i < length; i++) {
                if (str[i] == '"' && (i == 0 || str[i - 1] != '\\')) {
                    instring = !instring;
                }
                if (!instring) {
                    if (str[i] == ',' || str[i] == ']' || str[i] == '}') {
                        break;
                    }
                }
            }
            json.type = 0;
            if (i > 1 && str[i - 1] == '"' && str[0] == '"') {
                json.context = std::string(str + 1, i - 2);
            }
            else {
                json.context = std::string(str, i);
            }
            return i;
        }
    }
    return -1;
}

int loadconfig(sfp_t& sfp, std::string filename, std::string dir) {
    FILE* fi;
    char* ficon;
    int ficount;
    json_t conf;
    std::vector<char> jsonstr;
    if (!fopen_s(&fi, filename.c_str(), "rb")) {
        fseek(fi, 0, SEEK_END);
        ficount = ftell(fi);
        fseek(fi, 0, SEEK_SET);
        ficon = (char*)malloc(ficount * sizeof(char));
        ficount = fread(ficon, 1, ficount, fi);
        for (int i = 0; i < ficount; i++) {
            if (ficon[i] != ' ' && ficon[i] != '\n' && ficon[i] != '\r' && ficon[i] != '\t') {
                jsonstr.push_back(ficon[i]);
            }
        }
        int used;
        used = read(jsonstr.data(), jsonstr.size(), conf);
        fclose(fi);
        if (!fopen_s(&fi, (dir + "/config.txt").c_str(), "wb")) {
            fwrite(ficon, 1, ficount, fi);
            fclose(fi);
        }
        free(ficon);
    }


    for (int i = 0; i < conf.obj["atom"].vec.size(); i++) {
        sfp.tb.atoms.insert(std::make_pair(conf.obj["atom"].vec[i].context, i));
    }

    sscanf_s(conf.obj["kpoint"].context.c_str(), "%d", &sfp.tb.kp);
    sscanf_s(conf.obj["chemp"].context.c_str(), "%lf", &sfp.chemp);
    sscanf_s(conf.obj["invtemp"].context.c_str(), "%lf", &sfp.invtemp);
    sscanf_s(conf.obj["width"].context.c_str(), "%lf", &sfp.width);
    sscanf_s(conf.obj["u"].context.c_str(), "%lf", &sfp.u);
    sscanf_s(conf.obj["u_"].context.c_str(), "%lf", &sfp.u_);
    sscanf_s(conf.obj["j"].context.c_str(), "%lf", &sfp.j);
    sscanf_s(conf.obj["j_"].context.c_str(), "%lf", &sfp.j_);
    if (conf.obj.find("thread") != conf.obj.end()) {
        sscanf_s(conf.obj["thread"].context.c_str(), "%d", &sfp.thread);
    }
    else {
        sfp.thread = 4;
    }
    if (sfp.thread < 1) {
        sfp.thread = 1;
    }
    if (sfp.thread > 32) {
        sfp.thread = 32;
    }

    sfp.tb.atomn = sfp.tb.atoms.size();

    for (auto ht = conf.obj["hopping"].obj.begin(); ht != conf.obj["hopping"].obj.end(); ht++) {
        double value;
        sscanf_s(ht->second.obj["value"].context.c_str(), "%lf", &value);
        for (auto hp = ht->second.obj["pair"].vec.begin(); hp != ht->second.obj["pair"].vec.end(); hp++) {
            hopping_t hop;
            std::complex<double> phase;
            hop.from = sfp.tb.atoms[hp->vec[0].context];
            hop.to = sfp.tb.atoms[hp->vec[1].context];
            sscanf_s(hp->vec[2].context.c_str(), "%d", &hop.tx);
            sscanf_s(hp->vec[3].context.c_str(), "%d", &hop.ty);
            sscanf_s(hp->vec[4].context.c_str(), "%lf", phase._Val);
            sscanf_s(hp->vec[5].context.c_str(), "%lf", phase._Val + 1);
            hop.value = value * phase;
            sfp.tb.hoppings.push_back(hop);
        }
    }

    if (conf.obj.find("weight") != conf.obj.end()) {
        for (int i = 0; i < sfp.tb.atomn; i++) {
            double weight;
            sscanf_s(conf.obj["weight"].vec[i].context.c_str(), "%lf", &weight);
            sfp.qpweight.push_back(weight);
        }
    }
    else {
        for (int i = 0; i < sfp.tb.atomn; i++) {
            sfp.qpweight.push_back(1.0);
        }
    }
    return 0;
}

int init(sfp_t& sfp) {
    sfp.gridx = sfp.tb.kp;
    sfp.gridy = sfp.tb.kp;
    sfp.orbitalcount = sfp.tb.atomn;
    sfp.eigenvalue = new Eigen::VectorXd[sfp.gridx * sfp.gridy];
    sfp.eigenvaluecenter = new Eigen::VectorXd[sfp.gridx * sfp.gridy];
    sfp.eigenvector = new Eigen::MatrixXcd[sfp.gridx * sfp.gridy];
    return 0;
}

int getu(sfp_t& sfp) {
    sfp.us = Eigen::MatrixXcd(sfp.orbitalcount * sfp.orbitalcount, sfp.orbitalcount * sfp.orbitalcount);
    sfp.uc = Eigen::MatrixXcd(sfp.orbitalcount * sfp.orbitalcount, sfp.orbitalcount * sfp.orbitalcount);
    for (int l1 = 0; l1 < sfp.orbitalcount; l1++) {
        for (int l2 = 0; l2 < sfp.orbitalcount; l2++) {
            for (int l3 = 0; l3 < sfp.orbitalcount; l3++) {
                for (int l4 = 0; l4 < sfp.orbitalcount; l4++) {
                    if (l1 == l2 && l2 == l3 && l3 == l4) {
                        sfp.us(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = sfp.u;
                        sfp.uc(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = sfp.u;
                    }
                    else if (l1 == l3 && l2 == l4) {
                        sfp.us(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = sfp.u_;
                        sfp.uc(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = -sfp.u_ + 2.0 * sfp.j;
                    }
                    else if (l1 == l2 && l3 == l4) {
                        sfp.us(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = sfp.j;
                        sfp.uc(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = 2.0 * sfp.u_ - sfp.j;
                    }
                    else if (l1 == l4 && l2 == l3) {
                        sfp.us(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = sfp.j_;
                        sfp.uc(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = sfp.j_;
                    }
                    else {
                        sfp.us(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = 0;
                        sfp.uc(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3) = 0;
                    }
                }
            }
        }
    }
    return 0;
}

int gethm(sfp_t& sfp, double kx, double ky, Eigen::MatrixXcd& hm) {
    hm.setZero();
    if (sfp.tb.hoppings.size()) {
        for (int j = 0; j < sfp.tb.hoppings.size(); j++) {
            int f = sfp.tb.hoppings[j].from;
            int t = sfp.tb.hoppings[j].to;
            std::complex<double> h = sfp.tb.hoppings[j].value * std::complex<double>{cos(kx* sfp.tb.hoppings[j].tx + ky * sfp.tb.hoppings[j].ty), -sin(kx * sfp.tb.hoppings[j].tx + ky * sfp.tb.hoppings[j].ty)};
            hm(t, f) += h;
            hm(f, t) += std::conj(h);
        }
    }
    return 0;
}

int calceigenvalue(sfp_t& sfp,double kx,double ky, Eigen::VectorXd &eigenvalue, Eigen::MatrixXcd &eigenvector) {
    Eigen::MatrixXcd hm(sfp.tb.atomn, sfp.tb.atomn);
    gethm(sfp, kx, ky, hm);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hm);
    eigenvalue = solver.eigenvalues();
    eigenvector = solver.eigenvectors();
    return 0;
}

int calceigenvalue(sfp_t& sfp, double kx, double ky, Eigen::VectorXd& eigenvalue) {
    Eigen::MatrixXcd hm(sfp.tb.atomn, sfp.tb.atomn);
    gethm(sfp, kx, ky, hm);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hm, Eigen::EigenvaluesOnly);
    eigenvalue = solver.eigenvalues();
    return 0;
}


int calceigenvalue(sfp_t& sfp) {
    for (int kyn = 0; kyn < sfp.tb.kp; kyn++) {
        double ky = kyn * (2 * PI / sfp.gridx);
        for (int kxn = 0; kxn < sfp.tb.kp; kxn++) {
            double kx = kxn * (2 * PI / sfp.gridy);
            int index = kyn * sfp.gridx + kxn;
            calceigenvalue(sfp, kx, ky, sfp.eigenvalue[index], sfp.eigenvector[index]);
            calceigenvalue(sfp, kx + PI / sfp.tb.kp, ky + PI / sfp.tb.kp, sfp.eigenvaluecenter[index]);
        }
    }
    return 0;
}

double solve(sfp_t& sfp, double x0, double y0, double dy, double dx,int i,double level, int n,int flag) {
    double left = 0.0, right = 1.0;
    double x, y;
    Eigen::VectorXd eigenvalue;
    for (int it = 0; it < n; it++) {
        x = x0 + 0.5 * (left + right) * dx;
        y = y0 + 0.5 * (left + right) * dy; 
        calceigenvalue(sfp, x * (2 * PI / sfp.gridx), y * (2 * PI / sfp.gridx), eigenvalue);
        if ((eigenvalue(i) > level) == flag) {
            left= 0.5 * (left + right);
        }
        else {
            right = 0.5 * (left + right);
        }
    }
    return 0.5 * (left + right);
}

int getcontour(sfp_t& sfp, std::vector<double>& value, std::vector<double>& valuecenter, double level, int gridx, int gridy, std::vector<std::vector<surfacepoint_t>>& lines, int n) {
    std::vector<int> edgeoccu(gridx * gridy * 2, 0);
    std::vector<int> squaretype(gridx * gridy);
    for (int y = 0; y < gridy; y++) {
        int y1 = (y + 1) % gridy;
        for (int x = 0; x < gridx; x++) {
            int x1 = (x + 1) % gridx;
            squaretype[gridx * y + x] = (value[gridx * y + x] > level) + (value[gridx * y + x1] > level) * 2 + (value[gridx * y1 + x] > level) * 4 + (value[gridx * y1 + x1] > level) * 8;
            if (squaretype[gridx * y + x] == 6 && valuecenter[gridx * y + x] <= level) {
                squaretype[gridx * y + x] = 9;
            }
            if (squaretype[gridx * y + x] == 9 && valuecenter[gridx * y + x] <= level) {
                squaretype[gridx * y + x] = 6;
            }
        }
    }
    for (int y = 0; y < gridy; y++) {
        for (int x = 0; x < gridx; x++) {
            if (!edgeoccu[y * gridy + x]) {
                int x1 = (x + 1) % gridx;
                if ((value[y * gridx + x] - level) * (value[y * gridx + x1] - level) < 0) {
                    std::vector<surfacepoint_t> line;
                    int sqx = x, sqy = y, edge = 0;
                    int x0 = x, y0 = y;
                    while (!edgeoccu[y0 * gridx + x0 + (edge % 2) * gridx * gridy]) {
                        edgeoccu[y0 * gridx + x0 + (edge % 2) * gridx * gridy] = 1;

                        if (edge % 2) {
                            int y1 = (y0 + 1) % gridy;
                            double ky = y0 + solve(sfp, x0, y0, 1.0, 0.0, n, level, solveit, value[gridx * y0 + x0] > level);
                            double kx = x0;
                            line.push_back({ n,0,x0,y0,edge % 2,kx * (2 * PI / gridx),ky * (2 * PI / gridy) });
                        }
                        else {
                            int x1 = (x0 + 1) % gridx;
                            double kx = x0 + solve(sfp, x0, y0, 0.0, 1.0, n, level, solveit, value[gridx * y0 + x0] > level);
                            double ky = y0;
                            line.push_back({ n,0,x0,y0,edge % 2,kx * (2 * PI / gridx),ky * (2 * PI / gridy) });
                        }

                        sqx = (sqx + dirx[edge] + gridx) % gridx;
                        sqy = (sqy + diry[edge] + gridy) % gridy;
                        edge = connecttable[squaretype[sqy * gridy + sqx]][edge];
                        x0 = (sqx + (dirx[edge] + 1) / 2) % gridx;
                        y0 = (sqy + (diry[edge] + 1) / 2) % gridy;
                    }
                    lines.push_back(line);
                }
            }
        }
    }
    return 0;
}

int getfermi(sfp_t& sfp) {
    std::vector<double> value(sfp.gridx * sfp.gridy, 0.0), valuecenter(sfp.gridx * sfp.gridy, 0.0);
    for (int n = 0; n < sfp.orbitalcount; n++) {
        for (int y = 0; y < sfp.gridy; y++) {
            for (int x = 0; x < sfp.gridx; x++) {
                value[y * sfp.gridx + x] = sfp.eigenvalue[y * sfp.gridx + x](n);
                valuecenter[y * sfp.gridx + x] = sfp.eigenvaluecenter[y * sfp.gridx + x](n);
            }
        }
        getcontour(sfp, value, valuecenter, sfp.chemp, sfp.gridx, sfp.gridy, sfp.surface, n);
    }
    return 0;
}

int getstates(sfp_t& sfp) {
    for (int i = 0; i < sfp.surface.size(); i++) {
        for (int j = 0; j < sfp.surface[i].size(); j++) {
            state_t state;
            state.i = sfp.surface[i][j].i;
            state.kx = sfp.surface[i][j].kx;
            state.ky = sfp.surface[i][j].ky;
            state.nx = sfp.surface[i][j].nx;
            state.ny = sfp.surface[i][j].ny;
            int next = (j + 1) % sfp.surface[i].size();
            int last = (j - 1 + sfp.surface[i].size()) % sfp.surface[i].size();
            double dx = sfp.surface[i][next].kx - sfp.surface[i][last].kx + 2.0 * PI * ((sfp.surface[i][last].nx - sfp.surface[i][j].nx) / (sfp.gridx - 1) + (sfp.surface[i][j].nx - sfp.surface[i][next].nx) / (sfp.gridx - 1));
            double dy = sfp.surface[i][next].ky - sfp.surface[i][last].ky + 2.0 * PI * ((sfp.surface[i][last].ny - sfp.surface[i][j].ny) / (sfp.gridy - 1) + (sfp.surface[i][j].ny - sfp.surface[i][next].ny) / (sfp.gridy - 1));
            state.length = 0.5 * sqrt(dx * dx + dy * dy);

            Eigen::VectorXd eigenvalue;
            Eigen::MatrixXcd eigenvector;
            calceigenvalue(sfp, state.kx, state.ky, eigenvalue, eigenvector);
            state.eigenvector = eigenvector.col(state.i);

            double xp, xn, yp, yn;
            calceigenvalue(sfp, state.kx + 0.5 / sfp.gridx, state.ky, eigenvalue);
            xp = eigenvalue(state.i);
            calceigenvalue(sfp, state.kx - 0.5 / sfp.gridx, state.ky, eigenvalue);
            xn = eigenvalue(state.i);
            calceigenvalue(sfp, state.kx, state.ky + 0.5 / sfp.gridy, eigenvalue);
            yp = eigenvalue(state.i);
            calceigenvalue(sfp, state.kx, state.ky - 0.5 / sfp.gridy, eigenvalue);
            yn = eigenvalue(state.i);
            state.grad = sqrt((xp - xn) * (xp - xn) + (yp - yn) * (yp - yn));

            sfp.surface[i][j].stateindex = sfp.states.size();
            sfp.states.push_back(state);
        }
    }
    return 0;
}

double getenergypart(sfp_t& sfp, double e1, double e2) {
    if (abs(e2 - e1) * sfp.invtemp < 0.01) {
        double x = exp(0.5 * (e1 - sfp.chemp) * sfp.invtemp) + exp(-0.5 * (e1 - sfp.chemp) * sfp.invtemp);
        return -sfp.invtemp / (x * x);
    }
    else {
        double fe1 = 1.0 / (1.0 + exp((e1 - sfp.chemp) * sfp.invtemp));
        double fe2 = 1.0 / (1.0 + exp((e2 - sfp.chemp) * sfp.invtemp));
        return (fe2 - fe1) / (e2 - e1);
    }
}

int getnormalstatesusmat(sfp_t& sfp, double qx, double qy, Eigen::MatrixXcd& sus) {
    double kqx, kqy;
    Eigen::Matrix<std::complex<double>, -1, 1> col(sfp.orbitalcount * sfp.orbitalcount, 1);
    Eigen::Matrix<std::complex<double>, 1, -1> row(1, sfp.orbitalcount * sfp.orbitalcount);

    Eigen::MatrixXcd eigenvectorkq;
    Eigen::VectorXd eigenvaluekq;
    sus.setZero();
    for (int y = 0; y < sfp.gridy; y++) {
        kqy = y * (2 * PI / sfp.gridy) + qy;
        for (int x = 0; x < sfp.gridx; x++) {
            kqx = x * (2 * PI / sfp.gridx) + qx;
            int indexk = y * sfp.gridx + x;
            calceigenvalue(sfp, kqx, kqy, eigenvaluekq, eigenvectorkq);
            for (int i = 0; i < sfp.orbitalcount; i++) {
                for (int j = 0; j < sfp.orbitalcount; j++) {
                    double ek = sfp.eigenvalue[indexk](i);
                    double ekq = eigenvaluekq(j);
                    double energypart = getenergypart(sfp, ek, ekq);
                    for (int l2 = 0; l2 < sfp.orbitalcount; l2++) {
                        for (int l1 = 0; l1 < sfp.orbitalcount; l1++) {
                            col(l2 * sfp.orbitalcount + l1, 0) = std::conj(sfp.eigenvector[indexk](l2, i)) * eigenvectorkq(l1, j);
                        }
                    }

                    for (int l4 = 0; l4 < sfp.orbitalcount; l4++) {
                        for (int l3 = 0; l3 < sfp.orbitalcount; l3++) {
                            row(0, l4 * sfp.orbitalcount + l3) = sfp.eigenvector[indexk](l4, i) * std::conj(eigenvectorkq(l3, j));
                        }
                    }
                    sus += energypart * col * row;
                }
            }
        }
    }
    sus *= -1.0 / (sfp.gridx * sfp.gridy);
    return 0;
}

int getsusmat(sfp_t& sfp, int s, int e) {
    for (int i = s; i < e; i++) {
        double kx = sfp.pair[i].x * (2 * PI / sfp.gridx / solvediv);
        double ky = sfp.pair[i].y * (2 * PI / sfp.gridy / solvediv);
        Eigen::MatrixXcd sus(sfp.orbitalcount * sfp.orbitalcount, sfp.orbitalcount * sfp.orbitalcount);
        getnormalstatesusmat(sfp, kx, ky, sus);
        sfp.sus[i] = sus;
    }
    return 0;
}

DWORD WINAPI ms1(LPVOID data) {
    susmission_t* m = (susmission_t*)data;
    getsusmat(*(m->sfp), m->s, m->e);
    return 0;
}

int getsusmat(sfp_t& sfp, std::string dir) {
    for (int j = 0; j < sfp.states.size(); j++) {
        for (int i = j; i < sfp.states.size(); i++) {
            double kx = sfp.states[i].kx - sfp.states[j].kx;
            double ky = sfp.states[i].ky - sfp.states[j].ky;
            int x = floor(kx * (solvediv * sfp.gridx / 2 / PI) + 0.5);
            int y = floor(ky * (solvediv * sfp.gridy / 2 / PI) + 0.5);
            sfp.pairindex.insert(std::make_pair(int2{ x,y }, 0));
        }
    }
    sfp.pair.resize(sfp.pairindex.size());
    sfp.sus.resize(sfp.pairindex.size());
    int i = 0;
    for (auto& it : sfp.pairindex) {
        it.second = i;
        sfp.pair[i] = it.first;
        i++;
    }
    std::cout << sfp.pairindex.size() << "\n";

    FILE* fi;
    if (!fopen_s(&fi, (dir + "/sfptemp.data").c_str(), "rb")) {
        for (int i = 0; i < sfp.pairindex.size(); i++) {
            sfp.sus[i] = Eigen::MatrixXcd(sfp.orbitalcount * sfp.orbitalcount, sfp.orbitalcount * sfp.orbitalcount);
            fread(sfp.sus[i].data(), 1, sizeof(std::complex<double>) * sfp.sus[i].cols() * sfp.sus[i].rows(), fi);
        }
        fclose(fi);
    }
    else {
        susmission_t m[64];
        HANDLE thd[64];
        for (int i = 0; i < sfp.thread; i++) {
            m[i].sfp = &sfp;
            m[i].s = sfp.pair.size() * i / sfp.thread;
            m[i].e = sfp.pair.size() * (i + 1) / sfp.thread;
            thd[i] = CreateThread(NULL, 0, ms1, m + i, 0, NULL);
        }
        WaitForMultipleObjects(sfp.thread, thd, true, INFINITE);

        if (!fopen_s(&fi, (dir + "/sfptemp.data").c_str(), "wb")) {
            for (int i = 0; i < sfp.pairindex.size(); i++) {
                fwrite(sfp.sus[i].data(), 1, sizeof(std::complex<double>) * sfp.sus[i].cols() * sfp.sus[i].rows(), fi);
            }
            fclose(fi);
        }
    }
    return 0;
}

int getorbspacescat(sfp_t& sfp, double kx,double ky,Eigen::MatrixXcd &scatmat) {
    Eigen::MatrixXcd one(sfp.orbitalcount * sfp.orbitalcount, sfp.orbitalcount * sfp.orbitalcount);
    one.setZero();
    for (int i = 0; i < sfp.orbitalcount * sfp.orbitalcount; i++) {
        one(i, i) = 1;
    }
    
    Eigen::MatrixXcd col(sfp.orbitalcount* sfp.orbitalcount, sfp.orbitalcount* sfp.orbitalcount);
    Eigen::MatrixXcd row(sfp.orbitalcount* sfp.orbitalcount, sfp.orbitalcount* sfp.orbitalcount);
    col.setZero();
    row.setZero();
    for (int l2 = 0; l2 < sfp.orbitalcount; l2++) {
        for (int l1 = 0; l1 < sfp.orbitalcount; l1++) {
            col(l2 * sfp.orbitalcount + l1, l2 * sfp.orbitalcount + l1) = sfp.qpweight[l1] * sfp.qpweight[l2];
        }
    }
    for (int l4 = 0; l4 < sfp.orbitalcount; l4++) {
        for (int l3 = 0; l3 < sfp.orbitalcount; l3++) {
            row(l4 * sfp.orbitalcount + l3, l4 * sfp.orbitalcount + l3) = sfp.qpweight[l3] * sfp.qpweight[l4];
        }
    }

    int x = floor(kx * (solvediv * sfp.gridx / 2 / PI) + 0.5);
    int y = floor(ky * (solvediv * sfp.gridy / 2 / PI) + 0.5);

    
   /* Eigen::MatrixXcd sus(sfp.orbitalcount * sfp.orbitalcount, sfp.orbitalcount * sfp.orbitalcount);
    getnormalstatesusmat(sfp, kx, ky, sus); 
    sus = col * sus * row;
    
    std::cout << sus - col * sfp.sus[sfp.pairindex[{x, y}]] * row;
    */
    
    Eigen::MatrixXcd sus;
    if (sfp.pairindex.find({ x, y }) != sfp.pairindex.end()) {
        sus = col * sfp.sus[sfp.pairindex[{x, y}]] * row;
    }
    else {
        std::cout << "sus mat not found\n";
    }


    Eigen::MatrixXcd rpaspinsusmat = sus * (one - sfp.us * sus).inverse();
    Eigen::MatrixXcd rpachargesusmat = sus * (one + sfp.uc * sus).inverse();
    scatmat = 1.5 * sfp.us * rpaspinsusmat * sfp.us + 0.5 * sfp.us - 0.5 * sfp.uc * rpachargesusmat * sfp.uc + 0.5 * sfp.uc;
    return 0;
}

double getscat(sfp_t& sfp, state_t k1, state_t k2) {
    Eigen::MatrixXcd scatmat;
    double scat = 0;
    getorbspacescat(sfp, k1.kx - k2.kx, k1.ky - k2.ky, scatmat);

    for (int l1 = 0; l1 < sfp.orbitalcount; l1++) {
        for (int l2 = 0; l2 < sfp.orbitalcount; l2++) {
            for (int l3 = 0; l3 < sfp.orbitalcount; l3++) {
                for (int l4 = 0; l4 < sfp.orbitalcount; l4++) {
                    double weight = sfp.qpweight[l1] * sfp.qpweight[l2] * sfp.qpweight[l3] * sfp.qpweight[l4];
                    std::complex<double> m = std::conj(k1.eigenvector(l1)) * k1.eigenvector(l4) * k2.eigenvector(l2) * std::conj(k2.eigenvector(l3));
                    scat += weight * std::real(m * scatmat(l2 * sfp.orbitalcount + l1, l4 * sfp.orbitalcount + l3));
                }
            }
        }
    }
    return scat;
}

int getgapmat(sfp_t& sfp) {
    sfp.gapmat = Eigen::MatrixXd(sfp.states.size(), sfp.states.size());
    for (int j = 0; j < sfp.states.size(); j++) {
        for (int i = j; i < sfp.states.size(); i++) {
            double scat = getscat(sfp, sfp.states[i], sfp.states[j]);
            sfp.gapmat(i, j) = scat;
            sfp.gapmat(j, i) = scat;
        }
    }
    return 0;
}

int calcgap(sfp_t& sfp) {
    Eigen::MatrixXd weight(sfp.states.size(), sfp.states.size());
    weight.setZero();
    for (int i = 0; i < sfp.states.size(); i++) {
        weight(i, i) = sfp.states[i].length / sfp.states[i].grad;
    }
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(sfp.gapmat, weight, Eigen::ABx_lx | Eigen::ComputeEigenvectors);
    std::cout << solver.eigenvalues();
    sfp.gapfunc = solver.eigenvectors().col(0);
    return 0;
}


int main(int argc, char** argv) {
    std::string filename, dir;
    if (argc > 1) {
        filename = argv[1];
    }
    else {
        filename = "config.txt";
    }

    if (argc > 2) {
        dir = argv[2];
    }
    else {
        dir = "fese";
    }

    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);

    sfp_t sfp;
    loadconfig(sfp, filename, dir);
    init(sfp);
    getu(sfp);
    calceigenvalue(sfp);
    getfermi(sfp);
    getstates(sfp);
    
    getsusmat(sfp, dir);
    getgapmat(sfp);
    calcgap(sfp);

    FILE* fi;
    
    if (!fopen_s(&fi, (dir + "/band.csv").c_str(), "wb")) {
        for (int n = 0; n < sfp.orbitalcount; n++) {
            for (int j = 0; j < sfp.gridy; j++) {
                for (int i = 0; i < sfp.gridx; i++) {
                    fprintf(fi, "%f%c", sfp.eigenvalue[j * sfp.gridx + i](n), i < sfp.gridx - 1 ? ',' : '\n');
                }
            }
        }
        fclose(fi);
    }

    if (!fopen_s(&fi, (dir + "/gap.json").c_str(), "wb")) {
        fprintf(fi, "[");
        for (int i = 0; i < sfp.surface.size(); i++) {
            fprintf(fi, "[");
            int offx = 0, offy = 0;
            int j;
            for (j = 0; j < sfp.surface[i].size() - 1; j++) {
                fprintf(fi, "[%f,%f,%f],", sfp.surface[i][j].kx + offx * 2.0 * PI, sfp.surface[i][j].ky + offy * 2.0 * PI, sfp.gapfunc(sfp.surface[i][j].stateindex));
                offx += (sfp.surface[i][j].nx - sfp.surface[i][j + 1].nx) / (sfp.gridx - 1);
                offy += (sfp.surface[i][j].ny - sfp.surface[i][j + 1].ny) / (sfp.gridy - 1);
            }
            fprintf(fi, "[%f,%f,%f]]", sfp.surface[i][j].kx + offx * 2.0 * PI, sfp.surface[i][j].ky + offy * 2.0 * PI, sfp.gapfunc(sfp.surface[i][j].stateindex));
            if (i < sfp.surface.size() - 1) {
                fprintf(fi, ",");
            }
        }

        fprintf(fi, "]");
        fclose(fi);
    }
    
    if (!fopen_s(&fi, (dir + "/gapmat.csv").c_str(), "wb")) {
        for (int j = 0; j < sfp.states.size(); j++) {
            for (int i = 0; i < sfp.states.size(); i++) {
                fprintf(fi, "%f%c", sfp.gapmat(j, i), i < sfp.states.size() - 1 ? ',' : '\n');
            }
        }
        fclose(fi);
    }

    for (int i = 0; i < sfp.surface.size(); i++) {
        for (int j = 0; j < sfp.surface[i].size(); j++) {
            printf("%f,%f\n", sfp.surface[i][j].kx, sfp.surface[i][j].ky);
        }
        printf("\n");
    }
    return 0;
}
