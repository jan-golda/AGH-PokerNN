module Main where

  import NeuralNetwork
  import NeuralNetwork.IO
  import NeuralNetwork.Training
  import PokerNN
  
  
  -- | Reads randomly generated network from file, then trains on 1000 training cases, finally saves trained network to new file
  train :: IO ()
  train =
    readFile "./testNet.nn" >>= \netStr ->
    readFile "../data/training.data" >>= \dataStr ->
        let 
            network = stringToNetwork netStr
            trainingSet = stringToTrainingSet dataStr 10
            trainedNetwork = epochTraining network trainingSet 0.5 1
        --in putStrLn (show trainingSet)
        in writeNeuralNetwork "./testNet2.nn" trainedNetwork
        
        
  test :: IO ()
  test = 
      readFile "./testNet.nn" >>= \netStr ->
      getLine >>= \testStr ->
    let 
        network = stringToNetwork netStr
        test = stringToInput testStr
        out = feed network test
        message = outputToString out 
    in
        putStr message
            

  getRandom :: IO ()
  getRandom = randomNeuralNetwork 52 [10, 10] >>= \network -> writeNeuralNetwork "./testNet.nn" network