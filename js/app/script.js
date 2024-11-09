let weights = []

const minWeightChangeThreshold = 2.55; // Minimal weight change to stop training
let alfa = 0.002;
let teta = 0.23;

let dataForTrain = []

let weightChange = false

let btns = null

let b = []

window.addEventListener("load", () => {

    focused();

    fetch('trainData.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('File not found');
            }
            return response.json();
        })
        .then(data => {
            console.log("Data exists, returning 1");
            console.log("Weights:", data.weights);
            console.log("b:", data.b);

            weights = data.weights;
            b = data.b;
            test(weights, b)

            btnsAboutTrainSection.classList.add("hidden");
            btnSecSabt.classList.remove("hidden");
        })
        .catch(error => {
            console.log("No data found or error occurred, returning 0");
            console.error(error);

        });
});

function test(ws, bias) {
    let netInput = [0, 0];
    let counter = 0;

    fetch("testDataSets.json")
        .then(res => res.json())
        .then(array => {
            array.forEach(item => {
                item.data = item.data.flat();
            });

            array.forEach(item => {
                for (let j = 0; j < 2; j++) {
                    let sum = 0;
                    item.data.forEach((info, index) => {
                        sum += ws[index][j] * info;
                    });
                    netInput[j] = sum + bias[j];
                }

                let javab;
                if (netInput[0] > teta && netInput[1] < -teta) {
                    javab = [1, -1];
                } else if (netInput[0] < -teta && netInput[1] > teta) {
                    javab = [-1, 1];
                } else {
                    javab = [0, 0];
                }

                if (isArraysEqual(javab, item.y)) {
                    counter++;
                }
            });

            const accuracyValue = document.getElementById("accuracyValue");
            accuracyValue.innerHTML = `${((counter / array.length) * 100).toFixed(2)}%`;
        });
}

function isArraysEqual(arr1, arr2) {
    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i]) return false;
    }
    return true;
}

function showModal(message) {
    const modal = document.getElementById('blue-modal');
    const modalMessage = document.getElementById('modal-message');

    modalMessage.textContent = message;

    modal.style.display = 'block';

    modal.style.animation = 'fadeIn 0.5s ease';

    setTimeout(() => {
        closeModal(modal);
    }, 1500);
}

function closeModal(modal) {
    modal.style.animation = 'fadeOut 0.5s ease';
    modal.style.display = 'none';
}



const recognizeBtn = document.getElementById("recognizeBtn")

const btnsAboutTrainSection = document.getElementById("btnsAboutTrainSection")

const doneTrainBtn = document.getElementById("doneTrainBtn")

const btnSecSabt = document.getElementById("btnSecSabt")


function focused() {
    const btnContainer = document.getElementById("btnContainer");

    // Generate buttons and set their initial IDs
    for (let i = 0; i < 25; i++) {
        btnContainer.insertAdjacentHTML(
            "beforeend",
            `<button id="onactive" class="btn bg-blue-500 dark:bg-blue-600 hover:bg-blue-700 text-white font-semibold p-8 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"></button>`
        );
    }

    // Select all buttons
    btns = document.querySelectorAll(".btn");

    btns.forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.id === "onactive") {
                // Change classes for active state
                btn.classList.replace("bg-blue-500", "bg-rose-500");
                btn.classList.replace("dark:bg-blue-600", "dark:bg-rose-700");
                btn.classList.replace("hover:bg-blue-700", "hover:bg-rose-800");
                btn.id = "active"; // Change ID to active
            } else if (btn.id === "active") {
                // Revert classes to inactive state
                btn.classList.replace("bg-rose-500", "bg-blue-500");
                btn.classList.replace("dark:bg-rose-700", "dark:bg-blue-600");
                btn.classList.replace("hover:bg-rose-800", "hover:bg-blue-700");
                btn.id = "onactive"; // Change ID back to onactive
            }
        });
    });
}




doneTrainBtn.addEventListener("click", () => {

    fetch("trainDataSets.json").then(res => {
        if (res.ok) {
            return res.json()
        }
    }).then(array => {
        array.forEach(item => {
            item.data = item.data.flat()
        })
        dataForTrain = JSON.parse(JSON.stringify(array));;
        let yNetInput = null;

        let ykoll = 0;

        let index = null

        let epoch = 0

        let training = true

        for (let i = 0; i < 25; i++) {
            weights[i] = [Math.random() * 0.5, Math.random() * 0.5];
        }
        b = [Math.random() * 0.5, Math.random() * 0.5];


        for (let j = 0; j < 2; j++) {
            training = true;
            let totalWeightChange = 0; // Accumulate all weight changes for this epoch
            while (training) {
                totalWeightChange = 0; // Accumulate all weight changes for this epoch

                for (let item of dataForTrain) {
                    yNetInput = 0;
                    index = 0;
                    for (let x of item.data) {
                        yNetInput += weights[index][j] * x;
                        index++;
                    }
                    yNetInput += b[j];

                    if (yNetInput > teta) {
                        ykoll = 1;
                    } else if (yNetInput <= teta && yNetInput >= -teta) {
                        ykoll = 0;
                    } else {
                        ykoll = -1;
                    }

                    if (ykoll != item.y[j]) {
                        let i = 0;
                        item.data.forEach((x) => {
                            let deltaWeight = alfa * x * item.y[j];
                            weights[i][j] += deltaWeight;
                            totalWeightChange += Math.abs(deltaWeight); // Sum up the absolute weight changes
                            i++;
                        });
                        let deltaBias = alfa * item.y[j];
                        b[j] += deltaBias;
                        totalWeightChange += Math.abs(deltaBias); // Include bias change in the total
                    }
                }
                console.log(totalWeightChange);
                if (totalWeightChange < minWeightChangeThreshold) { // اگر هیچ تغییر وزنی نداشتیم، از حلقه خارج می‌شویم
                    training = false;
                }
                epoch++;
            }
        }

        console.log("finished at", epoch, "epoches");


        const data = {
            weights: weights,
            b: b
        };

        const jsonData = JSON.stringify(data);

        const blob = new Blob([jsonData], { type: 'application/json' });

        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'trainData.json';
        document.body.appendChild(a);
        a.click();

        document.body.removeChild(a);

        URL.revokeObjectURL(url);

        console.log("Data has been saved as JSON file!");

        btnsAboutTrainSection.classList.add("hidden");
        btnSecSabt.classList.remove("hidden");

    }).catch(err => {
        console.error(err)
    })

});





recognizeBtn.addEventListener("click", () => {
    let infoes = [];
    const flag = Array.from(btns).some(btn => btn.id === "active");
    let netInput = [0, 0];

    if (flag) {
        btns.forEach(btn => {
            if (btn.id === "active") {
                infoes.push(1);
            } else {
                infoes.push(-1);
            }
        });

        for (let j = 0; j < 2; j++) {
            let sum = 0;
            infoes.forEach((info, index) => {
                sum += weights[index][j] * info;
            });
            netInput[j] = sum + b[j];
        }
        console.log(netInput);
        if (netInput[0] > teta && netInput[1] < -teta) {
            showModal("it is a X");
        } else if (netInput[0] < -teta && netInput[1] > teta) {
            showModal("it is a O");
        } else {
            showModal("can't recognize :(");
        }

        setTimeout(() => {
            btns.forEach(btn => {
                if (btn.id == "active") {
                    // Revert classes to inactive state
                    btn.classList.replace("bg-rose-500", "bg-blue-500");
                    btn.classList.replace("dark:bg-rose-700", "dark:bg-blue-600");
                    btn.classList.replace("hover:bg-rose-800", "hover:bg-blue-700");
                    btn.id = "onactive"; // Change ID back to onactive
                }
            });
        }, 1500);
    } else {
        showModal("you should make a X or O first");
    }
});
