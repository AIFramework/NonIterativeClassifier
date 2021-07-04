using AI.DataStructs.Algebraic;
using AI.Extensions;
using AI.HightLevelFunctions;
using AI.ML.Classifiers;
using AI.ML.Datasets;
using AI.ML.NeuralNetwork.CoreNNW;
using AI.ML.NeuralNetwork.CoreNNW.Activations;
using AI.ML.NeuralNetwork.CoreNNW.Layers;
using AI.ML.NeuralNetwork.CoreNNW.Layers.Base;
using AI.Statistics;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NonIterative
{
    [Serializable]
    public class SimpleClassifier : IClassifier, IToLayer
    {
        private readonly Vector[] w;
        private readonly Vector bias;
        private readonly int numCl, inpDim;
        private readonly Func<Vector, Vector, double> ScalarProduct = AnalyticGeometryFunctions.ScalarProduct; // Скалярное произведеие

        /// <summary>
        /// Простой, однослойный, классификатор
        /// </summary>
        /// <param name="nClass">Число классов</param>
        /// <param name="inputDim">Размерность входа</param>
        public SimpleClassifier(int nClass, int inputDim)
        {
            numCl = nClass;
            inpDim = inputDim;

            w = new Vector[numCl]; // Векторы весов, 1 вектор - 1 класс
            for (int i = 0; i < numCl; i++)
            {
                w[i] = new Vector(inpDim);
            }

            bias = new Vector(numCl);// Векторы смещения
        }

        public int Classify(Vector inp)
        {
            return ClassifyProbVector(inp).IndexMax();
        }

        public Vector ClassifyProbVector(Vector inp)
        {
            double[] probs = new double[numCl];
            Parallel.For(0, numCl, i => probs[i] = ScalarProduct(inp, w[i]) + bias[i]);
            return Softmax(probs);
        }



        // Обучение
        public void Train(Vector[] features, int[] classes)
        {
            Vector[] classesV = new Vector[classes.Length];
            int maxInd = numCl - 1;
            Vector[] featuresV = new Vector[inpDim], classesVects = new Vector[numCl];
            Vector dispX = Statistic.EnsembleDispersion(features) + 1e-10; // Определение важности признака
            Vector stdX = dispX.TransformVector(Math.Sqrt);
            Vector meanX = Vector.Mean(features);


            for (int i = 0; i < classes.Length; i++)
            {
                classesV[i] = Vector.OneHotBePol(classes[i], maxInd); // Подготовка меток класса
            }

            for (int i = 0; i < inpDim; i++)
            {
                featuresV[i] = features.GetDimention(i);
            }

            for (int i = 0; i < numCl; i++)
            {
                classesVects[i] = classesV.GetDimention(i);
            }

            Parallel.For(0, numCl, i =>
            {
                for (int j = 0; j < inpDim; j++)
                {
                    w[i][j] = Statistic.Cov(featuresV[j], classesVects[i]);
                }

                w[i] = w[i] / dispX; // Масштабирование
                bias[i] = -ScalarProduct(meanX, w[i]); // Расчет весов смещения
            });
        }

        // Обучение
        public void Train(VectorIntDataset dataset)
        {
            Vector[] features = new Vector[dataset.Count];
            int[] classes = new int[dataset.Count];

            for (int i = 0; i < features.Length; i++)
            {
                classes[i] = dataset[i].ClassMark;
                features[i] = dataset[i].Features;
            }

            Train(features, classes);
        }

        // Преобразовать в слой сети
        public ILayer GetLayer()
        {
            FeedForwardLayer feedForwardLayer = new FeedForwardLayer(inpDim, numCl, new SoftmaxUnit(), new Random())
            {
                W = NeuralDataConverters.WVectorsToNNValue(w),
                Bias = new NNValue(bias)
            };
            return feedForwardLayer;
        }

        // Преобразовать в однослойную сеть
        public INetwork GetNetwork()
        {
            NNW net = new NNW
            {
                Layers = new List<ILayer>() { GetLayer() }
            };
            return net;
        }

        private Vector Softmax(Vector outNet)
        {
            Vector exp = outNet.TransformVector(Math.Exp);
            return outNet / exp.Sum();
        }

        public void Load(string path)
        {
            throw new NotImplementedException();
        }

        public void Save(string path)
        {
            throw new NotImplementedException();
        }

        ILayer IToLayer.GetLayer()
        {
            throw new NotImplementedException();
        }
    }
}
