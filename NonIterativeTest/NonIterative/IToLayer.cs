using AI.ML.NeuralNetwork.CoreNNW.Layers;

namespace NonIterative
{
    /// <summary>
    /// Поддержка конвертирования в слой
    /// </summary>
    public interface IToLayer
    {
        /// <summary>
        /// Конвертирование в слой
        /// </summary>
        ILayer GetLayer();

        /// <summary>
        /// Конвертирование в нейросеть
        /// </summary>
        INetwork GetNetwork();
    }
}
