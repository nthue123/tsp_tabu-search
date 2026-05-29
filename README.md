````markdown
# 🚗 tsp_tabu-search

Implementation of the **Tabu Search Algorithm** for solving the **Traveling Salesman Problem (TSP)** and route optimization problems.

This project applies a metaheuristic optimization technique to improve route quality and avoid local optimum solutions using a tabu memory structure.

---


# 🧠 About Tabu Search

Tabu Search is a local search optimization algorithm that enhances hill-climbing methods by:

- Avoiding repeated states
- Escaping local optimum traps
- Using adaptive memory structures
- Exploring broader solution spaces

### Main Components

- Initial solution generation
- Neighborhood exploration
- Tabu List management
- Aspiration criteria
- Best solution tracking

---

# ⚙️ Features

✅ Traveling Salesman Problem solver  
✅ Tabu Search implementation in C++  
✅ Nearest Neighbor initial solution  
✅ Dynamic neighborhood generation  
✅ Cost evaluation system  
✅ Tabu list optimization  
✅ Multiple dataset support  
✅ Workflow automation with GitHub Actions  

---

# 📂 Project Structure

```bash
tsp_tabu-search/
│
├── .github/
│   └── workflows/
│       └── main.yml          # GitHub Actions CI
│
├── move1020/
│   ├── bin/Debug/            # Compiled binaries
│   ├── obj/Debug/            # Object files
│   ├── data/                 # Input datasets
│   ├── workflow/             # Additional workflow resources
│   ├── main.cpp              # Main source code
│   ├── move1020.cbp          # CodeBlocks project file
│   ├── move1020.layout
│   └── move1020.depend
│
└── README.md
````

---

# 🔬 Algorithm Workflow

## 1. Generate Initial Solution

The algorithm starts with an initial route generated using:

* Nearest Neighbor Heuristic

---

## 2. Generate Neighbor Solutions

Neighbor solutions are created by:

* Swapping cities
* Reversing segments
* Reordering paths

---

## 3. Evaluate Candidate Solutions

Each candidate route is evaluated using:

```math
Total Cost = Σ Distance(city_i, city_{i+1})
```

---

## 4. Apply Tabu Restrictions

Previously visited moves are stored in the **Tabu List** to prevent cycling.

---

## 5. Update Best Solution

If a better solution is found:

* Update global best route
* Store new minimum cost

---

# 📈 Tabu Search Flow

```text
Initialize Solution
        ↓
Generate Neighborhood
        ↓
Evaluate Candidates
        ↓
Check Tabu List
        ↓
Select Best Candidate
        ↓
Update Tabu List
        ↓
Repeat Until Stopping Condition
```

---

# 💻 Technologies Used

| Technology     | Purpose                      |
| -------------- | ---------------------------- |
| C++            | Main programming language    |
| STL            | Data structures & algorithms |
| GitHub Actions | CI/CD automation             |
| CodeBlocks     | Development environment      |

---

# 🚀 Getting Started

## 1️⃣ Clone Repository

```bash
git clone https://github.com/nthue123/ParkingLot.git
cd tsp_tabu-search
```

---


# 📁 Input Data

Input datasets are stored in:

```bash
move1020/data/
```

The distance matrix is loaded and processed during runtime.

---

# 🧪 Performance

The Tabu Search algorithm significantly improves:

* Route optimization quality
* Escape from local optimum
* Runtime efficiency compared to brute-force search




```
```
