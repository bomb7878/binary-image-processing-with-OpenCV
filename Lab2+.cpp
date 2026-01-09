#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <map>
#include <windows.h>
#include <malloc.h>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;

// Структура для хранения контура с флагом отверстия
struct Contour {
    vector<Point> points;
    bool isHole;
    int contourIdx;
};

// Структура для графа минимального покрывающего дерева
struct Edge {
    int from, to;
    double weight;
    Point fromPt, toPt;

    bool operator>(const Edge& other) const { return weight > other.weight; }
};

// Функция поиска минимальной пары точек между двумя контурами
void findMinDistancePair(const vector<Point>& contour1, const vector<Point>& contour2,
    Point& pt1, Point& pt2, double& minDist, int maxSamples = 500) {
    minDist = (numeric_limits<double>::max)();

    // Субсемплирование для ускорения
    vector<Point> samples1 = contour1;
    vector<Point> samples2 = contour2;

    if (samples1.size() > maxSamples) {
        vector<Point> temp;
        int step = samples1.size() / maxSamples;
        for (size_t i = 0; i < samples1.size(); i += step) temp.push_back(samples1[i]);
        samples1 = temp;
    }

    if (samples2.size() > maxSamples) {
        vector<Point> temp;
        int step = samples2.size() / maxSamples;
        for (size_t i = 0; i < samples2.size(); i += step) temp.push_back(samples2[i]);
        samples2 = temp;
    }

    // Поиск минимального расстояния
    for (const Point& p1 : samples1) {
        for (const Point& p2 : samples2) {
            double dist = norm(p1 - p2);
            if (dist < minDist) {
                minDist = dist;
                pt1 = p1;
                pt2 = p2;
            }
        }
    }
}

// Функция поиска ближайшей точки на контуре
Point findNearestPointOnContour(const vector<Point>& contour, const Point& target) {
    double minDist = (numeric_limits<double>::max)();
    Point bestPoint;

    for (const Point& p : contour) {
        double dist = norm(p - target);
        if (dist < minDist) {
            minDist = dist;
            bestPoint = p;
        }
    }
    return bestPoint;
}

// Построение MST (алгоритм Прима) для контуров
vector<Edge> buildMST(const vector<Contour>& contours, int outerIdx) {
    int n = contours.size();
    vector<bool> inMST(n, false);
    vector<Edge> mstEdges;

    priority_queue<Edge, vector<Edge>, greater<Edge>> pq;
    inMST[outerIdx] = true;

    for (int i = 0; i < n; i++) {
        if (i != outerIdx) {
            Edge e;
            e.from = outerIdx;
            e.to = i;
            findMinDistancePair(contours[outerIdx].points, contours[i].points,
                e.fromPt, e.toPt, e.weight);
            pq.push(e);
        }
    }

    while (mstEdges.size() < n - 1 && !pq.empty()) {
        Edge e = pq.top();
        pq.pop();

        if (inMST[e.to]) continue;

        inMST[e.to] = true;
        mstEdges.push_back(e);

        for (int i = 0; i < n; i++) {
            if (!inMST[i] && i != e.to) {
                Edge newEdge;
                newEdge.from = e.to;
                newEdge.to = i;
                findMinDistancePair(contours[e.to].points, contours[i].points,
                    newEdge.fromPt, newEdge.toPt, newEdge.weight);
                pq.push(newEdge);
            }
        }
    }

    return mstEdges;
}

// Функция вставки контура с двумя разрезами (вход и выход)
void insertContourWithBridge(vector<Point>& mainContour, const vector<Point>& holeContour,
    const Point& entryPt, const Point& exitPt, const Point& holeEntryPt) {
    // Находим индексы точек в главном контуре
    int entryIdx = -1, exitIdx = -1;
    for (size_t i = 0; i < mainContour.size(); i++) {
        if (mainContour[i] == entryPt) entryIdx = i;
        if (mainContour[i] == exitPt) exitIdx = i;
    }

    if (entryIdx == -1 || exitIdx == -1) return;

    // Находим индекс точки входа на отверстии
    int holeEntryIdx = -1;
    for (size_t i = 0; i < holeContour.size(); i++) {
        if (holeContour[i] == holeEntryPt) {
            holeEntryIdx = i;
            break;
        }
    }
    if (holeEntryIdx == -1) return;

    vector<Point> newContour;
    newContour.reserve(mainContour.size() + holeContour.size() + 2);

    int current = entryIdx;
    while (current != exitIdx) {
        newContour.push_back(mainContour[current]);
        current = (current + 1) % mainContour.size();
    }
    newContour.push_back(mainContour[exitIdx]);

    newContour.push_back(holeEntryPt);

    // Добавляем отверстие полностью
    int holeIdx = holeEntryIdx;
    do {
        newContour.push_back(holeContour[holeIdx]);
        holeIdx = (holeIdx + 1) % holeContour.size();
    } while (holeIdx != holeEntryIdx);

    newContour.push_back(holeEntryPt);

    newContour.push_back(entryPt);

    // Оставшаяся часть главного контура
    current = exitIdx;
    while (current != entryIdx) {
        newContour.push_back(mainContour[current]);
        current = (current + 1) % mainContour.size();
    }

    mainContour = newContour;
}

// Основная функция обработки с MST
vector<Point> mergeContoursWithMST(vector<Contour>& contours) {
    if (contours.empty()) return {};

    // Находим внешний контур
    int outerIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        if (!contours[i].isHole) {
            outerIdx = i;
            break;
        }
    }
    if (outerIdx == -1) return {};

    vector<Edge> mstEdges = buildMST(contours, outerIdx);

    vector<vector<Edge>> adjList(contours.size());
    for (const Edge& e : mstEdges) {
        adjList[e.from].push_back(e);
        Edge reverse = e;
        reverse.from = e.to;
        reverse.to = e.from;
        reverse.fromPt = e.toPt;
        reverse.toPt = e.fromPt;
        adjList[e.to].push_back(reverse);
    }

    vector<Point> result = contours[outerIdx].points;
    vector<bool> visited(contours.size(), false);
    visited[outerIdx] = true;

    // Очередь для BFS: (индекс контура, его точки в главном контуре)
    queue<int> q;
    q.push(outerIdx);

    // Карта для хранения точек контуров в объединённом контуре
    map<int, vector<Point>> contourPointsInResult;
    contourPointsInResult[outerIdx] = result;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (const Edge& e : adjList[current]) {
            if (!visited[e.to]) {
                visited[e.to] = true;

                Point currentPt = findNearestPointOnContour(contourPointsInResult[current], e.fromPt);
                Point holePt = e.toPt;

                const vector<Point>& hole = contours[e.to].points;
                int holeEntryIdx = -1;
                for (size_t i = 0; i < hole.size(); i++) {
                    if (hole[i] == holePt) {
                        holeEntryIdx = i;
                        break;
                    }
                }

                if (holeEntryIdx == -1) continue;

                int oppositeIdx = (holeEntryIdx + hole.size() / 2) % hole.size();
                Point holeExitPt = hole[oppositeIdx];

                Point currentExitPt = findNearestPointOnContour(contourPointsInResult[current], holeExitPt);

                vector<Point> currentContourCopy = contourPointsInResult[current];
                insertContourWithBridge(currentContourCopy, hole, currentPt, currentExitPt, holePt);

                result = currentContourCopy;
                contourPointsInResult[e.to] = currentContourCopy;

                q.push(e.to);
            }
        }
    }

    return result;
}

// Основная функция обработки изображения
vector<vector<Point>> processImage(const Mat& binaryImage) {
    vector<vector<Point>> allContours;
    vector<Vec4i> hierarchy;

    // Находим все контуры с иерархией
    findContours(binaryImage, allContours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

    vector<vector<Contour>> objects;
    vector<bool> processed(allContours.size(), false);

    // Группируем контуры по объектам
    for (size_t i = 0; i < allContours.size(); i++) {
        if (processed[i]) continue;

        if (hierarchy[i][3] == -1) {
            vector<Contour> object;
            object.push_back({ allContours[i], false, (int)i });
            processed[i] = true;

            int childIdx = hierarchy[i][2];
            while (childIdx != -1) {
                object.push_back({ allContours[childIdx], true, childIdx });
                processed[childIdx] = true;
                childIdx = hierarchy[childIdx][0];
            }

            objects.push_back(object);
        }
    }

    vector<vector<Point>> resultContours;

    // Обрабатываем каждый объект
    for (auto& object : objects) {
        if (object.size() == 1) {
            resultContours.push_back(object[0].points);
        }
        else {
            vector<Point> merged = mergeContoursWithMST(object);
            resultContours.push_back(merged);
        }
    }

    return resultContours;
}

int main() {
    locale::global(locale("ru_RU.UTF-8"));
    wcout.imbue(locale());
    wcin.imbue(locale());
    wcerr.imbue(locale());
    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);

    string imagePath = "20.png";
    ifstream testFile(imagePath);

    if (!testFile.good()) {
        cerr << "ОШИБКА: Файл " << imagePath << " не найден в рабочей директории!" << endl;
        cerr << "Текущая рабочая директория может быть: " << filesystem::current_path() << endl;
        return -1;
    }
    testFile.close();

    Mat image = imread("20.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        wcerr << L"Не удалось загрузить изображение!" << endl;
        return -1;
    }

    Mat binary;
    threshold(image, binary, 127, 255, THRESH_BINARY);

    auto start = chrono::high_resolution_clock::now();
    vector<vector<Point>> result = processImage(binary);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    wcout << L"Обработка заняла: " << duration.count() << L" мс" << endl;
    wcout << L"Найдено объектов: " << result.size() << endl;

    Mat visualization = Mat::zeros(binary.size(), CV_8UC3);
    RNG rng(12345);

    for (size_t i = 0; i < result.size(); i++) {
        Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(visualization, result, i, color, 1, LINE_AA);
    }

    imwrite("result.png", visualization);

    if (!result.empty() && !result[0].empty()) {
        wcout << L"Первые 10 точек первого контура:" << endl;
        for (int i = 0; i < min(10, (int)result[0].size()); i++) {
            cout << result[0][i] << endl;
        }
    }

    return 0;
}