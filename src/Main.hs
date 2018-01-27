module Main where

  import NeuralNetwork
  import NeuralNetwork.IO
  import NeuralNetwork.Training
  import PokerNN
  
  
  -- | Reads randomly generated network from file, then trains on 1000 training cases, finally saves trained network to new file
  main :: IO ()
  main =
    readFile "./randomNet.nn" >>= \netStr ->
    readFile "../data/training.data" >>= \dataStr ->
        let 
            network = stringToNetwork netStr
            trainingSet = stringToTrainingSet dataStr 1000
            trainedNetwork = epochTraining network trainingSet 0.3 100
        in writeNeuralNetwork "./trainedNet.nn" network
            
