using AI;
using AI.ML.Classifiers;
using AI.ML.Datasets;
using AI.HightLevelFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AI.Statistics;

namespace NonIterative
{
    [Serializable]
    public class SimpleClassifier : IClassifier
    {

        Vector[] w;
        int numCl, inpDim;
        Vector meanX;

        Func<Vector, Vector, double> ScalarProduct = AnalyticGeometryFunctions.ScalarProduct;

        public SimpleClassifier(int nClass, int inputDim) 
        {
            numCl = nClass;
            inpDim = inputDim;

            w = new Vector[numCl]; // Векторы весов, 1 вектор - 1 класс
            for (int i = 0; i < numCl; i++) w[i] = new Vector(inpDim);
            meanX = new Vector(inpDim);
        }

        public int Classify(Vector inp)
        {
            return ClassifyProbVector(inp).IndexMax();
        }

        public Vector ClassifyProbVector(Vector inp)
        {
            double[] probs = new double[numCl];
            Parallel.For(0,numCl, i => probs[i] = ScalarProduct(inp - meanX, w[i])); 
            return Softmax(probs);
        }

        public void Load(string path)
        {
            throw new NotImplementedException();
        }

        public void Save(string path)
        {
            throw new NotImplementedException();
        }

        public void Train(Vector[] features, int[] classes)
        {
            Vector[] classesV = new Vector[classes.Length];
            int maxInd = numCl - 1;
            Vector[] featuresV = new Vector[inpDim], classesVects = new Vector[numCl];
            Vector dispX = Statistic.EnsembleDispersion(features)+1e-10; // Определение важности признака
            Vector stdX = dispX.TransformVector(Math.Sqrt); 
            meanX = Vector.Mean(features);


            for (int i = 0; i < classes.Length; i++) 
                classesV[i] = Vector.OneHotBePol(classes[i], maxInd); // Подготовка меток класса

            for (int i = 0; i < inpDim; i++)
                featuresV[i] = features.GetDimention(i);

            for (int i = 0; i < numCl; i++)
                classesVects[i] = classesV.GetDimention(i);


            Parallel.For(0, numCl, i =>
            {
                for (int j = 0; j < inpDim; j++)
                    w[i][j] = Statistic.CorrelationCoefficient(featuresV[j], classesVects[i]);

                w[i] = (w[i] / stdX).GetUnitVector() / dispX; // Нормализация по важности признаков
            });
        }

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

        private Vector Softmax(Vector outNet) 
        {
            Vector exp = outNet.TransformVector(Math.Exp);
            return outNet / exp.Sum();
        }
    }
}
