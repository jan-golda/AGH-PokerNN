module Main where

  import NeuralNetwork
  import NeuralNetwork.IO
  import NeuralNetwork.Training
  import PokerNN


  -- | Reads network from file and trains on given TrainingSet, finally saves network to outputFile
  train :: String -> String -> Int -> Int -> Double -> String -> IO ()
  train networkFile dataFile cases epochs learningRate outputFile =
    readFile networkFile >>= \netStr ->
    readFile dataFile >>= \dataStr ->
        let
            network = fromString netStr
            trainingSet = stringToTrainingSet dataStr cases
            trainedNetwork = epochTraining network trainingSet learningRate epochs
        in writeFile outputFile (toString trainedNetwork)


  test :: String -> String -> IO ()
  test networkFile testStr =
      readFile networkFile >>= \netStr ->
    let
        network = fromString netStr
        test = stringToInput testStr
        out = feed network test
        message = outputToString out
    in
        putStr message


  getRandom :: [Int] -> String -> IO ()
  getRandom layersInfo outputFile = randomNeuralNetwork 52 (layersInfo ++ [10]) >>= \network -> writeFile outputFile (toString network)
