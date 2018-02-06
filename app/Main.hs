module Main where

  import NeuralNetwork
  import NeuralNetwork.IO as IO
  import NeuralNetwork.Training
  import PokerNN
  import System.Random.Shuffle as Shuffle
  
  
  -- | Main function
  main :: IO ()
  main = putStrLn "type 'help' to see quick usage guide" >> mainLoop [] []
       
       
  -- | Main function loop creating console interface
  mainLoop :: NeuralNetwork -> TrainingSet -> IO ()
  mainLoop network set = putStr "> " >> getLine >>= \command ->
    case command of
         
         "" -> mainLoop network set 
         
         "help" -> putStrLn "Usage:"
            >> putStrLn "\t'loadNet' - loads network from file"
            >> putStrLn "\t'loadSet' - loads training set from file"
            >> putStrLn "\t'random' - generates random network"
            >> putStrLn "\t'train' - trains network on previously loaded training set"
            >> putStrLn "\t'test' - tests network on user's input"
            >> putStrLn "\t'save' - saves network to file"
            >> putStrLn "\t'quit' - exits program" >> mainLoop network set
         
         "quit" -> putStrLn "goodbye"
         
         "loadNet" -> putStr "network file path: "
            >> getLine >>= \filePath
            -> readFile filePath 
            >>= \netStr ->
                let
                    newNet = IO.fromString netStr
                in
                    mainLoop newNet set
                     
         "loadSet" -> putStr "data file path: "
            >> getLine >>= \filePath
            -> putStr "cases number: "
            >> getLine >>= \numStr ->
                let
                    num = read numStr :: Int
                in
                    readFile filePath >>= \setStr ->
                        let
                            newSet = PokerNN.stringToTrainingSet setStr num
                        in
                            mainLoop network newSet
                
         "test" -> putStr "input: "
            >> getLine >>= \testStr -> 
                let
                    test = stringToInput testStr 
                    output = feed network test 
                    result = outputToString output
                in
                    putStrLn result >> mainLoop network set
                
         "save" -> putStr "file name: "
            >> getLine >>= \filePath
            -> writeFile filePath (IO.toString network)
            >> mainLoop network set
            
         "train" -> putStr "batch size (will be chosen randomly from loaded training set): "
            >> getLine >>= \batchSizeStr
            -> putStr "epochs number: "
            >> getLine >>= \epochsStr
            -> putStr "learning rate: "
            >> getLine >>= \rateStr ->
                let
                    batchSize = read batchSizeStr :: Int
                    epochsNumber = read epochsStr :: Int
                    learningRate = read rateStr :: Double
                in
                    getRandomTrainingBatch set batchSize >>= \batch ->
                        let
                            newNet = epochTraining network batch learningRate epochsNumber
                        in
                            mainLoop newNet set
            
         "random" -> putStr "hidden layer parameters: "
             >> getLine
             >>= \paramStr ->
                let
                    parameters = map (read :: String -> Int) $ words paramStr
                in
                    randomNeuralNetwork 52 (parameters ++ [10]) >>= \newNet
                    -> mainLoop newNet set
                    
                
         _ -> putStrLn "unknown command" >> mainLoop network set
            
  
  
  
        
  -- | Takes random subset of given size from given TrainingSet
  getRandomTrainingBatch :: TrainingSet -> Int -> IO TrainingSet
  getRandomTrainingBatch trainingSet batchSize = Shuffle.shuffleM trainingSet >>= \shuffledSet -> return $ take batchSize shuffledSet

  -- | Generates random neural network with 
  getRandom :: [Int] -> IO NeuralNetwork
  getRandom layersInfo = randomNeuralNetwork 52 (layersInfo ++ [10])
