#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

// ============================================================
// 1. KHAI BÁO BIẾN TOÀN CỤC & CẤU TRÚC DỮ LIỆU
// ============================================================

int N = 0;
vector<vector<double>> distMatrix;
vector<vector<int>> tabuMatrix;

struct City {
    int id;
    double x, y;
};
vector<City> cities;

// ============================================================
// 2. CÁC HÀM HỖ TRỢ
// ============================================================

double euclidean_distance(const City& c1, const City& c2) {
    return sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2));
}

void precomputeDistances() {
    distMatrix.assign(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            distMatrix[i][j] = euclidean_distance(cities[i], cities[j]);
        }
    }
}

void readInput(const string& filename) {
    ifstream inFile(filename);
    if (!inFile) { cerr << "LỖI: Không thể mở tệp " << filename << endl; exit(1); }

    string line;
    string edgeType = "EXPLICIT";
    bool inSection = false;
    cities.clear();
    vector<double> weights;

    while (getline(inFile, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        if (line.empty()) continue;

        if (line.find("DIMENSION") != string::npos) {
            size_t num_start = line.find_first_of("0123456789");
            if (num_start != string::npos) N = stoi(line.substr(num_start));
        }
        else if (line.find("EDGE_WEIGHT_TYPE") != string::npos) {
            if (line.find("EUC_2D") != string::npos) edgeType = "EUC_2D";
            else if (line.find("EXPLICIT") != string::npos) edgeType = "EXPLICIT";
        }
        else if (line.find("NODE_COORD_SECTION") != string::npos) { inSection = true; continue; }
        else if (line.find("EDGE_WEIGHT_SECTION") != string::npos) { inSection = true; continue; }
        else if (line.find("EOF") != string::npos) break;

        if (inSection) {
            stringstream ss(line);
            if (edgeType == "EUC_2D") {
                int id; double x, y;
                if (ss >> id >> x >> y) cities.push_back({id - 1, x, y});
            } else {
                double val; while (ss >> val) weights.push_back(val);
            }
        }
    }
    inFile.close();

    if (N == 0) { cerr << "Lỗi DIMENSION!" << endl; exit(1); }

    if (edgeType == "EUC_2D") {
        if (cities.size() < N) N = cities.size();
        precomputeDistances();
    } else {
        distMatrix.assign(N, vector<double>(N, 0.0));
        int k = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (k < weights.size()) distMatrix[i][j] = distMatrix[j][i] = weights[k++];
            }
        }
    }
}

double calculateTourCost(const vector<int>& tour) {
    double cost = 0.0;
    for (int i = 0; i < N; ++i) {
        cost += distMatrix[tour[i]][tour[(i + 1) % N]];
    }
    return cost;
}

vector<int> createNearestNeighborTour() {
    vector<int> tour;
    vector<bool> visited(N, false);
    int cur = 0;
    tour.push_back(cur);
    visited[cur] = true;

    for (int i = 0; i < N - 1; ++i) {
        double min_d = 1e9;
        int next = -1;
        for (int j = 0; j < N; ++j) {
            if (!visited[j] && distMatrix[cur][j] < min_d) {
                min_d = distMatrix[cur][j];
                next = j;
            }
        }
        if (next == -1) break;
        cur = next;
        tour.push_back(cur);
        visited[cur] = true;
    }
    return tour;
}

void diversifyTour(vector<int>& tour, int strength) {
    for (int i = 0; i < strength; ++i) {
        int p1 = rand() % N;
        int p2 = rand() % N;
        if (p1 != p2) swap(tour[p1], tour[p2]);
    }
}

// ============================================================
// 3. THUẬT TOÁN TABU SEARCH
// ============================================================

void runTabuSearch(string instanceName) {
    auto start_time = chrono::high_resolution_clock::now();

    const int MAX_ITERATIONS = 2000;
    const int TABU_TENURE = (N > 0) ? N/2 : 10;
    const int DIVERSIFY_THRESHOLD = 150;

    // --- MỞ FILE LOG ---
    size_t lastdot = instanceName.find_last_of(".");
    string rawName = (lastdot == string::npos) ? instanceName : instanceName.substr(0, lastdot);

    // Tách tên file ra khỏi đường dẫn để đặt tên log
    size_t lastSlash = rawName.find_last_of("/\\");
    string pureName = (lastSlash == string::npos) ? rawName : rawName.substr(lastSlash + 1);

    string logFilename = "TS_LOG_" + pureName + ".txt";
    ofstream logFile(logFilename);
    bool logEnabled = logFile.is_open();

    vector<int> currentTour = createNearestNeighborTour();
    double currentCost = calculateTourCost(currentTour);
    if (currentCost < 0) currentCost = abs(currentCost);

    vector<int> bestTour = currentTour;
    double bestCost = currentCost;
    int bestIter = 0;

    tabuMatrix.assign(N, vector<int>(N, 0));
    int noImprovement = 0;

    // --- GHI TIÊU ĐỀ BẢNG VÀO FILE LOG ---
    if (logEnabled) {
        logFile << "--- TABU SEARCH LOG FOR INSTANCE: " << instanceName << " ---" << endl;
        logFile << "\n" << left
             << setw(8) << "Iter"
             << setw(15) << "Current_Cost"
             << setw(15) << "Best_Cost"
             << setw(10) << "New_Best"
             << setw(12) << "Move_Type"
             << setw(20) << "Move" << endl;
        logFile << string(80, '-') << endl;
    }

    // --- Vòng lặp chính ---
    for (int k = 1; k <= MAX_ITERATIONS; ++k) {
        noImprovement++;
        int moveType = rand() % 2;
        double bestDelta = 1e9;
        int best_i = -1, best_j = -1;
        bool moveFound = false;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;

                double delta = 0;
                bool isTabu = false;

                // Move 1-0
                if (moveType == 0) {
                    if (i == j || i == (j + 1) % N) continue;
                    int C = currentTour[i];
                    int A = currentTour[(i - 1 + N) % N];
                    int B = currentTour[(i + 1) % N];
                    int X = currentTour[j];
                    int Y = currentTour[(j + 1) % N];

                    double rem = distMatrix[A][C] + distMatrix[C][B] + distMatrix[X][Y];
                    double add = distMatrix[A][B] + distMatrix[X][C] + distMatrix[C][Y];
                    delta = add - rem;

                    if (tabuMatrix[A][C] > k || tabuMatrix[C][B] > k) isTabu = true;
                }
                // Move 2-0
                else if (moveType == 1) {
                    if (i >= N - 1) continue;
                    int i_next = (i + 1) % N;
                    if (j == i || j == i_next || j == (i - 1 + N) % N || j == (i + 2) % N) continue;
                    int C = currentTour[i];
                    int D = currentTour[i_next];
                    int A = currentTour[(i - 1 + N) % N];
                    int B = currentTour[(i + 2) % N];
                    int X = currentTour[j];
                    int Y = currentTour[(j + 1) % N];

                    double rem = distMatrix[A][C] + distMatrix[D][B] + distMatrix[X][Y];
                    double add = distMatrix[A][B] + distMatrix[X][C] + distMatrix[D][Y];
                    delta = add - rem;

                    if (tabuMatrix[A][C] > k || tabuMatrix[D][B] > k) isTabu = true;
                }

                if (currentCost + delta > 0) {
                    bool aspiration = (currentCost + delta < bestCost);

                    if (delta < bestDelta) {
                        if (!isTabu || aspiration) {
                            bestDelta = delta;
                            best_i = i;
                            best_j = j;
                            moveFound = true;
                        }
                    }
                }
            }
        }

        // --- THỰC HIỆN BƯỚC ĐI TỐT NHẤT VÀ CẬP NHẬT TRẠNG THÁI ---
        if (moveFound) {
            int A, B;
            string moveDescStr = "";
            string moveTypeStr = "";
            string nextBestStr = "No";

            // Cập nhật tour và Tabu List
            if (moveType == 0) {
                moveTypeStr = "1-0 (Ins)";
                int id_C = currentTour[best_i];
                int id_X = currentTour[best_j];
                moveDescStr = to_string(id_C + 1) + "-" + to_string(id_X + 1);

                A = currentTour[(best_i - 1 + N) % N];
                B = currentTour[(best_i + 1) % N];

                tabuMatrix[A][id_C] = tabuMatrix[id_C][A] = k + TABU_TENURE;
                tabuMatrix[id_C][B] = tabuMatrix[B][id_C] = k + TABU_TENURE;

                currentTour.erase(currentTour.begin() + best_i);
                auto it = find(currentTour.begin(), currentTour.end(), id_X);
                int new_pos = distance(currentTour.begin(), it);
                currentTour.insert(currentTour.begin() + new_pos + 1, id_C);
            }
            else if (moveType == 1) {
                moveTypeStr = "2-0 (Blk)";
                int i = best_i;
                int id_C = currentTour[i];
                int id_D = currentTour[i + 1];
                int id_X = currentTour[best_j];
                moveDescStr = "[" + to_string(id_C + 1) + "," + to_string(id_D + 1) + "]-" + to_string(id_X + 1);

                A = currentTour[(i - 1 + N) % N];
                int B = currentTour[(i + 2) % N];

                tabuMatrix[A][id_C] = tabuMatrix[id_C][A] = k + TABU_TENURE;
                tabuMatrix[id_D][B] = tabuMatrix[B][id_D] = k + TABU_TENURE;

                currentTour.erase(currentTour.begin() + i, currentTour.begin() + i + 2);
                auto it = find(currentTour.begin(), currentTour.end(), id_X);
                int pos_X = distance(currentTour.begin(), it);
                currentTour.insert(currentTour.begin() + pos_X + 1, id_D);
                currentTour.insert(currentTour.begin() + pos_X + 1, id_C);
            }

            currentCost += bestDelta;

            if (currentCost < bestCost) {
                bestCost = currentCost;
                bestTour = currentTour;
                bestIter = k;
                noImprovement = 0;
                nextBestStr = "Yes";
            }

            // --- GHI DÒNG LOG BẢNG VÀ TABU LIST VÀO FILE LOG ---
            if (logEnabled) {
                logFile << left
                        << setw(8) << k
                        << setw(15) << fixed << setprecision(2) << currentCost
                        << setw(15) << fixed << setprecision(2) << bestCost
                        << setw(10) << nextBestStr
                        << setw(12) << moveTypeStr
                        << setw(20) << moveDescStr << endl;

                // Ghi Tabu List
                logFile << "    |_ Tabu: ";
                int countTabu = 0;
                bool printed = false;
                for(int r=0; r<N; ++r) {
                    for(int c=r+1; c<N; ++c) {
                        if(tabuMatrix[r][c] > k) {
                            logFile << "(" << r+1 << "-" << c+1 << ":" << tabuMatrix[r][c] << ") ";
                            countTabu++;
                            printed = true;
                            if(countTabu >= 8) goto end_tabu_loop;
                        }
                    }
                }
                end_tabu_loop:;
                if(countTabu >= 8) logFile << "...";
                if(!printed) logFile << "EMPTY";
                logFile << endl;
            }
        }

        // --- DIVERSIFICATION ---
        if (noImprovement >= DIVERSIFY_THRESHOLD) {
            if (logEnabled) {
                logFile << string(80, '-') << endl;
                logFile << "!!! DIVERSIFYING (SHUFFLE) at Iteration " << k << " !!!" << endl;
                logFile << "Current Cost before Diversification: " << fixed << setprecision(2) << currentCost << endl;
                logFile << string(80, '-') << endl;
            }
            diversifyTour(currentTour, N);
            currentCost = calculateTourCost(currentTour);
            tabuMatrix.assign(N, vector<int>(N, 0));
            noImprovement = 0;
        }
    }

    // --- ĐÓNG FILE LOG ---
    if (logEnabled) {
        logFile.close();
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    // --- BẮT ĐẦU VÙNG REPORT (Dùng để trích xuất) ---
    cout << "!!!REPORT_START!!!" << endl;

    // --- FINAL RESULT REPORT (In ra màn hình console) ---
    cout << "\n========================================" << endl;
    cout << "          FINAL RESULT REPORT           " << endl;
    cout << "========================================" << endl;

    cout << "Instance    : " << pureName << endl;
    cout << "Runtime     : " << fixed << setprecision(4) << elapsed.count() << " s" << endl;
    cout << "Best Iter   : " << bestIter << endl;
    cout << "Tabu Tenure : " << TABU_TENURE << endl;
    cout << "Objective   : " << fixed << setprecision(2) << bestCost << endl;
    cout << "Solution    : [";
    for(int i = 0; i < N; ++i) {
        cout << bestTour[i] + 1;
        if (i < N - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "========================================" << endl;

    // --- KẾT THÚC VÙNG REPORT (Dùng để trích xuất) ---
    cout << "!!!REPORT_END!!!" << endl;

    if (logEnabled) {
        // Thông báo cho người dùng biết file log đã được tạo (in sau REPORT_END)
        cout << "\n✅ Log chi tiết đã được lưu vào file: " << logFilename << endl;
    }
}

// ============================================================
// 4. HÀM MAIN (Đã sửa để nhận đối số dòng lệnh)
// ============================================================
int main(int argc, char* argv[]) {
    srand(unsigned(time(0)));

    // KIỂM TRA ĐỐI SỐ DÒNG LỆNH
    if (argc != 2) {
        cerr << "LỖI SỬ DỤNG: Cú pháp: ./tsp_solver <tên_file_tsp>" << endl;
        cerr << "Ví dụ: ./tsp_solver data/eil51.tsp" << endl;
        return 1;
    }

    // Lấy tên file từ đối số dòng lệnh đầu tiên (argv[1])
    string filename = argv[1];

    readInput(filename);

    if (N > 0) runTabuSearch(filename);

    return 0;
}
