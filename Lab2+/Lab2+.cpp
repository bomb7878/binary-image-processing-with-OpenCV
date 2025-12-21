#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;

// Класс для оптимизированного поиска ближайших точек
class PointSearch {
private:
    struct Node {
        Point point;
        int axis;
        Node* left;
        Node* right;

        Node(const Point& p, int a) : point(p), axis(a), left(nullptr), right(nullptr) {}
    };

    Node* root;

    Node* buildTree(vector<Point>& points, int depth) {
        if (points.empty()) return nullptr;

        int axis = depth % 2;
        auto cmp = [axis](const Point& a, const Point& b) {
            return axis == 0 ? a.x < b.x : a.y < b.y;
            };

        size_t median = points.size() / 2;
        nth_element(points.begin(), points.begin() + median, points.end(), cmp);

        Node* node = new Node(points[median], axis);

        vector<Point> leftPoints(points.begin(), points.begin() + median);
        vector<Point> rightPoints(points.begin() + median + 1, points.end());

        node->left = buildTree(leftPoints, depth + 1);
        node->right = buildTree(rightPoints, depth + 1);

        return node;
    }

    void nearestNeighbor(Node* node, const Point& target, double& bestDist, Point& bestPoint) const {
        if (!node) return;

        double dist = norm(node->point - target);
        if (dist < bestDist) {
            bestDist = dist;
            bestPoint = node->point;
        }

        int axis = node->axis;
        double diff = axis == 0 ? target.x - node->point.x : target.y - node->point.y;

        Node* first = diff < 0 ? node->left : node->right;
        Node* second = diff < 0 ? node->right : node->left;

        nearestNeighbor(first, target, bestDist, bestPoint);

        if (fabs(diff) < bestDist) {
            nearestNeighbor(second, target, bestDist, bestPoint);
        }
    }

public:
    PointSearch(const vector<Point>& points) {
        if (points.empty()) {
            root = nullptr;
            return;
        }
        vector<Point> pts = points;
        root = buildTree(pts, 0);
    }

    Point findNearest(const Point& target) const {
        double bestDist = numeric_limits<double>::max();
        Point bestPoint;
        nearestNeighbor(root, target, bestDist, bestPoint);
        return bestPoint;
    }
};

// Структура для хранения контура с информацией о его типе
struct ContourInfo {
    vector<Point> points;
    bool isHole;
    PointSearch* searchTree;

    ContourInfo(const vector<Point>& pts, bool hole) : points(pts), isHole(hole), searchTree(nullptr) {
        if (!points.empty()) {
            searchTree = new PointSearch(points);
        }
    }

    ~ContourInfo() {
        delete searchTree;
    }

    // Найти ближайшую точку на этом контуре к заданной точке
    Point findNearestPoint(const Point& p) const {
        if (!searchTree) return p;
        return searchTree->findNearest(p);
    }
};

// Найти минимальное расстояние между двумя контурами и соответствующие точки
void findClosestPoints(const ContourInfo& contour1, const ContourInfo& contour2,
    Point& bestP1, Point& bestP2, double& minDist) {
    minDist = numeric_limits<double>::max();

    // Для оптимизации используем субсемплирование если контуры слишком большие
    const int maxSamples = 500;

    vector<Point> samples1 = contour1.points;
    vector<Point> samples2 = contour2.points;

    if (samples1.size() > maxSamples) {
        vector<Point> temp;
        int step = samples1.size() / maxSamples;
        for (size_t i = 0; i < samples1.size(); i += step) {
            temp.push_back(samples1[i]);
        }
        samples1 = temp;
    }

    if (samples2.size() > maxSamples) {
        vector<Point> temp;
        int step = samples2.size() / maxSamples;
        for (size_t i = 0; i < samples2.size(); i += step) {
            temp.push_back(samples2[i]);
        }
        samples2 = temp;
    }

    for (const Point& p1 : samples1) {
        Point p2 = contour2.findNearestPoint(p1);
        double dist = norm(p1 - p2);
        if (dist < minDist) {
            minDist = dist;
            bestP1 = p1;
            bestP2 = p2;
        }
    }
}

// Найти индекс точки в контуре
int findPointIndex(const vector<Point>& contour, const Point& p) {
    for (size_t i = 0; i < contour.size(); i++) {
        if (contour[i] == p) return i;
    }
    return -1;
}

// Объединить внешний контур с внутренними с минимальными разрезами
vector<Point> mergeContoursWithHoles(ContourInfo& outer, vector<ContourInfo>& holes) {
    if (holes.empty()) return outer.points;

    struct CutInfo {
        Point outerPoint;
        Point holePoint;
        double distance;
        int holeIndex;

        bool operator<(const CutInfo& other) const {
            return distance < other.distance;
        }
    };

    vector<CutInfo> cuts;

    // Находим оптимальные разрезы для каждого отверстия
    for (size_t i = 0; i < holes.size(); i++) {
        Point outerPoint, holePoint;
        double distance;

        findClosestPoints(outer, holes[i], outerPoint, holePoint, distance);
        cuts.push_back({ outerPoint, holePoint, distance, (int)i });
    }

    sort(cuts.begin(), cuts.end());
    vector<Point> result = outer.points;

    for (const auto& cut : cuts) {
        const auto& hole = holes[cut.holeIndex];

        // Находим индексы точек разреза в текущем результате и в отверстии
        int outerIdx = findPointIndex(result, cut.outerPoint);
        int holeIdx = findPointIndex(hole.points, cut.holePoint);

        if (outerIdx == -1 || holeIdx == -1) continue;

        vector<Point> newResult;
        newResult.reserve(result.size() + hole.points.size() + 2);

        for (int i = 0; i <= outerIdx; i++) {
            newResult.push_back(result[i]);
        }

        newResult.push_back(cut.holePoint);

        for (size_t i = 1; i <= hole.points.size(); i++) {
            int idx = (holeIdx + i) % hole.points.size();
            newResult.push_back(hole.points[idx]);
        }

        newResult.push_back(cut.holePoint);

        for (size_t i = outerIdx + 1; i < result.size(); i++) {
            newResult.push_back(result[i]);
        }

        result = newResult;
    }

    return result;
}

// Основная функция обработки
vector<vector<Point>> processImage(const Mat& binaryImage) {
    vector<vector<Point>> allContours;
    vector<Vec4i> hierarchy;

    // Находим все контуры с иерархией
    findContours(binaryImage, allContours, hierarchy,
        RETR_CCOMP, CHAIN_APPROX_NONE);

    vector<vector<ContourInfo>> objects;

    // Проходим по иерархии и группируем контуры по объектам
    vector<bool> processed(allContours.size(), false);

    for (size_t i = 0; i < allContours.size(); i++) {
        if (processed[i]) continue;

        if (hierarchy[i][3] == -1) {
            vector<ContourInfo> object;

            object.emplace_back(allContours[i], false);
            processed[i] = true;

            // Ищем все внутренние контуры (отверстия) этого объекта
            int childIdx = hierarchy[i][2];
            while (childIdx != -1) {
                object.emplace_back(allContours[childIdx], true);
                processed[childIdx] = true;
                childIdx = hierarchy[childIdx][0];
            }

            objects.push_back(object);
        }
    }

    vector<vector<Point>> resultContours;

    for (auto& object : objects) {
        if (object.size() == 1) {
            // Нет отверстий - просто добавляем внешний контур
            resultContours.push_back(object[0].points);
        }
        else {
            // Есть отверстия - объединяем
            ContourInfo& outer = object[0];
            vector<ContourInfo> holes(object.begin() + 1, object.end());

            vector<Point> merged = mergeContoursWithHoles(outer, holes);
            resultContours.push_back(merged);
        }
    }

    return resultContours;
}

int main() {
    Mat image = imread("D:/C++ Projects/Lab2+/Lab2+/20.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Не удалось загрузить изображение!" << endl;
        return -1;
    }

    Mat binary;
    threshold(image, binary, 127, 255, THRESH_BINARY);

    // Обработка с замером времени
    auto start = chrono::high_resolution_clock::now();

    vector<vector<Point>> result = processImage(binary);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Обработка заняла: " << duration.count() << " мс" << endl;
    cout << "Найдено объектов: " << result.size() << endl;

    Mat visualization = Mat::zeros(binary.size(), CV_8UC3);
    RNG rng(12345);

    for (size_t i = 0; i < result.size(); i++) {
        Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(visualization, result, i, color, 1, LINE_AA);
    }

    imwrite("result.png", visualization);

    // Вывод контуров
    if (!result.empty()) {
        cout << "Пример контура (первые 10 точек):" << endl;
        for (int i = 0; i < min(10, (int)result[0].size()); i++) {
            cout << "(" << result[0][i].x << ", " << result[0][i].y << ")" << endl;
        }
    }

    return 0;
}