module NeuralNetwork (Neuron, Layer) where

  data Neuron = Neuron {bias :: Double, weights :: [Double]} deriving (Show)

  type Layer = [Neuron]

  type NeuralNetwork = [Layer]
