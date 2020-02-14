async function run(){
    const csvUrl = 'js/iris.csv';
    const trainingData = tf.data.csv(csvUrl, {
        columnConfigs: {
            species: {
                isLabel: true
            }
        }
    });
    
    const numOfFeatures = (await trainingData.columnNames()).length - 1;
    const numOfSamples = 150;

    const convertedData = trainingData.map(({xs, ys}) => {

        const labels = [ys.species == "setosa" ? 1 : 0,
                        ys.species == "virginica" ? 1 : 0,
                        ys.species == "versicolor" ? 1 : 0]
        return{ xs: Object.values(xs), ys: Object.values(labels)};   
    }).batch(10);

    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "sigmoid", units: 5}));
    model.add(tf.layers.dense({activation: "softmax", units: 3}));
    model.compile({loss: "categoricalCrossentropy", optimizer: tf.train.adam(0.06)});

    await model.fitDataset(convertedData, {

        epochs:10,
        callbacks:{ onEpochEnd: async (epoch, logs) =>{
                    console.log("Epoch: " + epoch + " Loss: " + logs.loss);}}
    });

    const testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1,4]);
    const prediction = model.predict(testVal);
    alert(prediction);
    // const pIndex = tf.argMax(prediction, axis = 1).dataSync();
    // const classNames = ["Setosa", "Virginica", "Versicolor"];
            
    
    // alert(classNames[pIndex])
}

run();