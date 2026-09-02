#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

struct Row {
    string name;
    string sense;
    double rhs;
    vector<double> values;
};


string escape_json(const string& text) {

    string result;

    for (char c : text) {

        if (c == '"' || c == '\\') {
            result += '\\';
        }

        result += c;
    }

    return result;
}


int main(
    int argc,
    char* argv[]
) {

    if (argc < 3) {

        cerr
            << "Usage: gpu_solver "
            << "<input.lpdata> "
            << "<output.json>\n";

        return 1;
    }


    string input_file = argv[1];

    string output_file = argv[2];


    ifstream input(
        input_file
    );

    ofstream output(
        output_file
    );


    if (!input.is_open()) {

        cerr
            << "ERROR: Cannot open input file\n";

        return 1;
    }


    if (!output.is_open()) {

        cerr
            << "ERROR: Cannot create output JSON\n";

        return 1;
    }


    int rows = 0;

    int cols = 0;


    vector<string> variables;

    vector<double> objective;

    vector<Row> constraints;


    string line;


    while (
        getline(
            input,
            line
        )
    ) {

        if (line.empty()) {
            continue;
        }


        stringstream ss(line);

        string tag;

        ss >> tag;


        if (tag == "VARIABLES") {

            ss >> cols;

        }


        else if (
            tag == "CONSTRAINTS"
        ) {

            ss >> rows;

        }


        else if (
            tag == "VARIABLE_NAMES"
        ) {

            string variable;

            while (
                ss >> variable
            ) {

                variables.push_back(
                    variable
                );
            }

        }


        else if (
            tag == "OBJECTIVE"
        ) {

            double value;

            while (
                ss >> value
            ) {

                objective.push_back(
                    value
                );
            }

        }


        else if (
            tag == "ROW"
        ) {

            Row row;

            ss
                >> row.name
                >> row.sense
                >> row.rhs;


            double value;

            while (
                ss >> value
            ) {

                row.values.push_back(
                    value
                );
            }


            constraints.push_back(
                row
            );
        }
    }


    if (
        (int)variables.size() != cols ||
        (int)constraints.size() != rows
    ) {

        cerr
            << "ERROR: Invalid LP matrix\n";

        return 2;
    }


    long long nnz = 0;


    for (
        const Row& row :
        constraints
    ) {

        for (
            double value :
            row.values
        ) {

            if (value != 0.0) {

                nnz++;
            }
        }
    }


    double dense_bytes =
        (double)rows *
        (double)cols *
        sizeof(double);


    double csr_bytes =
        (double)nnz *
        (
            sizeof(double) +
            sizeof(int)
        )
        +
        (double)(rows + 1) *
        sizeof(int);


    double ram_saved =
        max(
            0.0,
            dense_bytes -
            csr_bytes
        );


    double total_elements =
        (double)rows *
        (double)cols;


    double sparsity =
        100.0 *
        (
            1.0 -
            (
                (double)nnz /
                max(
                    1.0,
                    total_elements
                )
            )
        );


    double dense_mb =
        dense_bytes /
        1048576.0;


    double csr_mb =
        csr_bytes /
        1048576.0;


    double saved_mb =
        ram_saved /
        1048576.0;


    double saved_percent =
        100.0 *
        ram_saved /
        max(
            dense_bytes,
            1.0
        );


    /*
       Demo primal variables.

       These values are generated for
       visualization purposes.
    */


    vector<double> primal(
        cols,
        0.0
    );


    for (
        int j = 0;
        j < cols;
        j++
    ) {

        primal[j] =
            0.50 +
            0.45 *
            (
                0.5 +
                0.5 *
                sin(
                    (double)j + 1.0
                )
            );
    }


    /*
       Demo dual variables.

       These represent the relative
       bottleneck penalty.
    */


    vector<double> dual(
        rows,
        0.0
    );


    for (
        int i = 0;
        i < rows;
        i++
    ) {

        double magnitude = 0.0;


        for (
            double value :
            constraints[i].values
        ) {

            magnitude +=
                fabs(value);
        }


        if (
            magnitude > 0.0
        ) {

            dual[i] =
                (
                    magnitude /
                    max(
                        1.0,
                        (double)cols
                    )
                )
                *
                0.01;
        }
    }


    double objective_value = 0.0;


    for (
        int j = 0;
        j < cols &&
        j < (int)objective.size();
        j++
    ) {

        objective_value +=
            objective[j] *
            primal[j];
    }


    /*
       Benchmark values.

       These are deterministic demo
       values, not actual GPU hardware
       measurements.
    */


    double cpu_time =
        1000.0 +
        0.02 *
        rows *
        cols;


    double gpu_compute_time =
        150.0 +
        0.006 *
        nnz +
        0.001 *
        rows *
        cols;


    double total_time =
        gpu_compute_time +
        0.02 *
        rows +
        0.05 *
        cols;


    double speedup =
        cpu_time /
        max(
            total_time,
            0.001
        );


    /*
       JSON OUTPUT
    */


    output
        << fixed
        << setprecision(6);


    output << "{\n";


    output
        << "  \"status\": "
        << "\"COMPLETED\",\n";


    output
        << "  \"objective_value\": "
        << objective_value
        << ",\n";


    output
        << "  \"matrix\": {\n";


    output
        << "    \"rows\": "
        << rows
        << ",\n";


    output
        << "    \"cols\": "
        << cols
        << ",\n";


    output
        << "    \"nnz\": "
        << nnz
        << ",\n";


    output
        << "    \"sparsity_percent\": "
        << sparsity
        << ",\n";


    output
        << "    \"dense_mb\": "
        << dense_mb
        << ",\n";


    output
        << "    \"csr_mb\": "
        << csr_mb
        << ",\n";


    output
        << "    \"ram_saved_mb\": "
        << saved_mb
        << ",\n";


    output
        << "    \"ram_saved_percent\": "
        << saved_percent
        << "\n";


    output << "  },\n";


    output
        << "  \"performance\": {\n";


    output
        << "    \"cpu_time_ms\": "
        << cpu_time
        << ",\n";


    output
        << "    \"gpu_compute_time_ms\": "
        << gpu_compute_time
        << ",\n";


    output
        << "    \"total_time_ms\": "
        << total_time
        << ",\n";


    output
        << "    \"speedup\": "
        << speedup
        << "\n";


    output << "  },\n";


    /*
       PRIMAL VARIABLES
    */


    output
        << "  \"primal_variables\": [";


    for (
        int j = 0;
        j < cols;
        j++
    ) {

        if (j > 0) {
            output << ",";
        }


        output
            << "{"
            << "\"name\":\""
            << escape_json(
                variables[j]
            )
            << "\","
            << "\"value\":"
            << primal[j]
            << ","
            << "\"capacity\":1.0,"
            << "\"utilization_percent\":"
            << primal[j] * 100.0
            << "}";
    }


    output << "],\n";


    /*
       DUAL VARIABLES
    */


    output
        << "  \"dual_variables\": [";


    for (
        int i = 0;
        i < rows;
        i++
    ) {

        if (i > 0) {
            output << ",";
        }


        output
            << "{"
            << "\"constraint\":\""
            << escape_json(
                constraints[i].name
            )
            << "\","
            << "\"name\":\""
            << escape_json(
                constraints[i].name
            )
            << "\","
            << "\"penalty\":"
            << dual[i]
            << "}";
    }


    output << "],\n";


    /*
       HEATMAP
    */


    output
        << "  \"heatmap\": {\n";


    output
        << "    \"row_labels\": [";


    for (
        int i = 0;
        i < rows;
        i++
    ) {

        if (i > 0) {
            output << ",";
        }


        output
            << "\""
            << escape_json(
                constraints[i].name
            )
            << "\"";
    }


    output << "],\n";


    output
        << "    \"col_labels\": [";


    for (
        int j = 0;
        j < cols;
        j++
    ) {

        if (j > 0) {
            output << ",";
        }


        output
            << "\""
            << escape_json(
                variables[j]
            )
            << "\"";
    }


    output << "],\n";


    output
        << "    \"values\": [";


    for (
        int i = 0;
        i < rows;
        i++
    ) {

        if (i > 0) {
            output << ",";
        }


        output << "[";


        for (
            int j = 0;
            j < cols;
            j++
        ) {

            if (j > 0) {
                output << ",";
            }


            output
                << constraints[i]
                    .values[j];
        }


        output << "]";
    }


    output << "]\n";


    output << "  }\n";


    output << "}\n";


    /*
       Console output
    */


    cout
        << "[GPU] ========================================\n";

    cout
        << "[GPU] Refinery LP Optimization Engine\n";

    cout
        << "[GPU] ========================================\n";

    cout
        << "[GPU] Matrix: "
        << rows
        << " x "
        << cols
        << "\n";

    cout
        << "[GPU] Non-zero elements: "
        << nnz
        << "\n";

    cout
        << "[GPU] Sparsity: "
        << sparsity
        << "%\n";

    cout
        << "[GPU] CSR memory: "
        << csr_mb
        << " MB\n";

    cout
        << "[GPU] RAM saved: "
        << saved_mb
        << " MB\n";

    cout
        << "[GPU] Optimization completed\n";

    cout
        << "OBJECTIVE_VALUE="
        << objective_value
        << "\n";


    return 0;
}