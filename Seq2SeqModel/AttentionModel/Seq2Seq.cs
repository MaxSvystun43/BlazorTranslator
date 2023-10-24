using Seq2SeqModel.Entities;
using Seq2SeqModel.Extentions;
using Seq2SeqModel.Model;

namespace Seq2SeqModel.AttentionModel
{
    [Serializable]
    public class Seq2Seq
    {

        public event EventHandler IterationDone;

        public int max_word = 100; // max length of generated sentences 
        public Dictionary<string, int> wordToIndex = new Dictionary<string, int>();
        public Dictionary<int, string> indexToWord = new Dictionary<int, string>();
        public List<string> vocab = new List<string>();
        public List<List<string>> InputSequences;
        public List<List<string>> OutputSequences;
        public int HiddenSize;
        public int wordSize;

        // optimization  hyperparameters
        public double RegularizationStrength = 0.000001; // L2 regularization strength
        public double learning_rate = 0.001; // learning rate
        public double clipval = 5.0; // clip gradients at this value


        public RMSPropOptimazer solver;
        public Weights Embedding;
        public Encoder encoder;
        public Encoder ReversEncoder;
        public AttentionDecoder decoder;

        public bool UseDropout { get; set; }


        //Output Layer Weights
        public Weights WeightsOutput { get; set; }
        public Weights BiasOutput { get; set; }

        public int Depth { get; set; }

        public Seq2Seq(int inputSize, int hiddenSize, int depth, List<List<string>> input, List<List<string>> output, bool useDropout)
        {
            InputSequences = input;
            OutputSequences = output;
            Depth = depth;
            // list of sizes of hidden layers
            wordSize = inputSize; // size of word embeddings.
            UseDropout = useDropout;

            HiddenSize = hiddenSize;
            solver = new RMSPropOptimazer();

            OneHotEncoding(input, output);

            WeightsOutput = new Weights(hiddenSize, vocab.Count + 2, true);
            BiasOutput = new Weights(1, vocab.Count + 2, 0);

            Embedding = new Weights(vocab.Count + 2, wordSize, true);

            encoder = new Encoder(hiddenSize, wordSize, depth);
            ReversEncoder = new Encoder(hiddenSize, wordSize, depth);

            decoder = new AttentionDecoder(hiddenSize, wordSize, depth);
        }

        public Seq2Seq(int inputSize, int hiddenSize, int depth, bool useDropout)
        {
            Depth = depth;
            // list of sizes of hidden layers
            wordSize = inputSize; // size of word embeddings.

            HiddenSize = hiddenSize;
            solver = new RMSPropOptimazer();

            WeightsOutput = new Weights(hiddenSize, vocab.Count + 2, true);
            BiasOutput = new Weights(1, vocab.Count + 2, 0);

            Embedding = new Weights(vocab.Count + 2, wordSize, true);

            encoder = new Encoder(hiddenSize, wordSize, depth);
            ReversEncoder = new Encoder(hiddenSize, wordSize, depth);

            decoder = new AttentionDecoder(hiddenSize, wordSize, depth);
        }

        public void Train(int trainingEpoch)
        {
            for (int ep = 0; ep < trainingEpoch; ep++)
            {
                Random r = new Random();
                for (int itr = 0; itr < InputSequences.Count; itr++)
                {
                    // sample sentence from data
                    List<string> OutputSentence;
                    GraphComputations graph;
                    double cost;
                    List<Weights> encoded = new List<Weights>();
                    Encode(r, out OutputSentence, out graph, out cost, encoded);
                    cost = DecodeOutput(OutputSentence, graph, cost, encoded);

                    graph.BackPropogation();
                    UpdateParameters();
                    Reset();
                    if (IterationDone != null)
                    {
                        IterationDone(this, new CostEvent()
                        {
                            Cost = cost / OutputSentence.Count,
                            Iteration = ep
                        });
                    }
                }
            }
        }


        public List<string> Predict(List<string> inputSeq)
        {
            ReversEncoder.Reset();
            encoder.Reset();
            decoder.Reset();

            List<string> result = new List<string>();

            var graph = new GraphComputations(false);

            List<string> revseq = inputSeq.ToList();
            revseq.Reverse();
            List<Weights> encoded = new List<Weights>();
            for (int i = 0; i < inputSeq.Count; i++)
            {
                int index = wordToIndex[inputSeq[i]];
                int indexReverse = wordToIndex[revseq[i]];
                var weight = graph.GetByRow(Embedding, index);
                var encoderWeight = encoder.Encode(weight, graph);
                var weightReverse = graph.GetByRow(Embedding, indexReverse);
                var encoderReverse = ReversEncoder.Encode(weightReverse, graph);

                var encoderData = graph.AddColumns(encoderWeight, encoderReverse);

                encoded.Add(encoderData);
            }

            var indexInput = 1;
            while (true)
            {
                var decoderWeight = graph.GetByRow(Embedding, indexInput);
                var decoderData = decoder.Decode(decoderWeight, encoded, graph);

                if (UseDropout)
                {
                    for (int i = 0; i < decoderData.Weight.Length; i++)
                    {
                        decoderData.Weight[i] *= 0.2;
                    }
                }
                var tempGraph = graph.AddWeights(graph.Multiply(decoderData, WeightsOutput), BiasOutput);

                if (UseDropout)
                {
                    for (int i = 0; i < tempGraph.Weight.Length; i++)
                    {
                        tempGraph.Weight[i] *= 0.2;
                    }
                }
                var probs = graph.SoftmaxWithCrossEntropy(tempGraph);
                var maxValue = probs.Weight[0];
                var maxIndex = 0;

                for (int i = 1; i < probs.Weight.Length; i++)
                {
                    if (probs.Weight[i] > maxValue)
                    {
                        maxValue = probs.Weight[i];
                        maxIndex = i;
                    }
                }
                var pred = maxIndex;

                if (pred == 0) break; // END token predicted, break out

                if (result.Count > max_word) { break; } // something is wrong 
                var letter2 = indexToWord[pred];
                result.Add(letter2);
                indexInput = pred;
            }

            return result;
        }

        public ModelData Save()
        {
            ModelData saveModel = new ModelData();
            saveModel.BiasOut = BiasOutput;
            saveModel.Clipval = clipval;
            saveModel.Decoder = decoder;
            saveModel.Depth = Depth;
            saveModel.Encoder = encoder;
            saveModel.HiddenSizes = HiddenSize;
            saveModel.LearningRate = learning_rate;
            saveModel.LetterSize = wordSize;
            saveModel.MaxSentenceGen = max_word;
            saveModel.RegularizationStrength = RegularizationStrength;
            saveModel.ReversEncoder = ReversEncoder;
            saveModel.UseDropout = UseDropout;
            saveModel.WeightsOut = WeightsOutput;
            saveModel.WeightsIn = Embedding;
            saveModel.wordToIndex = wordToIndex;
            saveModel.indexToWord = indexToWord;

            return saveModel;
        }
        public void Load(ModelData model)
        {
            BiasOutput = model.BiasOut;
            clipval = model.Clipval;
            decoder = model.Decoder;
            Depth = model.Depth;
            encoder = model.Encoder;
            HiddenSize = model.HiddenSizes;
            learning_rate = model.LearningRate;
            wordSize = model.LetterSize;
            max_word = 100;
            RegularizationStrength = model.RegularizationStrength;
            ReversEncoder = model.ReversEncoder;
            UseDropout = model.UseDropout;
            WeightsOutput = model.WeightsOut;
            Embedding = model.WeightsIn;
            wordToIndex = model.wordToIndex;
            indexToWord = model.indexToWord;
        }

        private void OneHotEncoding(List<List<string>> input, List<List<string>> output)
        {
            // count up all words
            Dictionary<string, int> dict = new Dictionary<string, int>();
            wordToIndex = new Dictionary<string, int>();
            indexToWord = new Dictionary<int, string>();
            vocab = new List<string>();
            for (int j = 0, n2 = input.Count; j < n2; j++)
            {
                var item = input[j];
                for (int i = 0, n = item.Count; i < n; i++)
                {
                    var tempWord = item[i];
                    if (dict.Keys.Contains(tempWord)) { dict[tempWord] += 1; }
                    else { dict.Add(tempWord, 1); }
                }

                var item2 = output[j];
                for (int i = 0, n = item2.Count; i < n; i++)
                {
                    var tempWord = item2[i];
                    if (dict.Keys.Contains(tempWord)) { dict[tempWord] += 1; }
                    else { dict.Add(tempWord, 1); }
                }

            }

            var index = 2;
            foreach (var word in dict)
            {
                if (word.Value >= 1)
                {
                    // add word to vocab
                    wordToIndex[word.Key] = index;
                    indexToWord[index] = word.Key;
                    vocab.Add(word.Key);
                    index++;
                }
            }
        }

        private void Encode(Random rand, out List<string> OutputSentence, out GraphComputations graph, out double cost, List<Weights> encoded)
        {
            var sentIndex = rand.Next(0, InputSequences.Count);
            var inputSentence = InputSequences[sentIndex];
            var reverseSentence = InputSequences[sentIndex].ToList();
            reverseSentence.Reverse();
            OutputSentence = OutputSequences[sentIndex];
            graph = new GraphComputations();

            cost = 0.0;

            for (int i = 0; i < inputSentence.Count; i++)
            {
                int index = wordToIndex[inputSentence[i]];
                int indexRevers = wordToIndex[reverseSentence[i]];
                var weightInIndex = graph.GetByRow(Embedding, index);
                var eOutput = encoder.Encode(weightInIndex, graph);
                var weightInIndexRevers = graph.GetByRow(Embedding, indexRevers);
                var eOutput2 = ReversEncoder.Encode(weightInIndexRevers, graph);
                encoded.Add(graph.AddColumns(eOutput, eOutput2));
            }
        }

        private double DecodeOutput(List<string> OutputSentence, GraphComputations graph, double cost, List<Weights> encoded)
        {
            int indexInput = 1;
            for (int i = 0; i < OutputSentence.Count + 1; i++)
            {
                int indexTarget = 0;
                if (i == OutputSentence.Count)
                {
                    indexTarget = 0;
                }
                else
                {
                    indexTarget = wordToIndex[OutputSentence[i]];
                }


                var weightInIndex = graph.GetByRow(Embedding, indexInput);
                var eOutput = decoder.Decode(weightInIndex, encoded, graph);
                if (UseDropout)
                {
                    eOutput = graph.Dropout(eOutput, 0.2);
                }
                var output = graph.AddWeights(
                       graph.Multiply(eOutput, WeightsOutput), BiasOutput);
                if (UseDropout)
                {
                    output = graph.Dropout(output, 0.2);
                }

                var probs = graph.SoftmaxWithCrossEntropy(output);

                cost += -Math.Log(probs.Weight[indexTarget]);

                output.Gradient = probs.Weight;
                output.Gradient[indexTarget] -= 1;
                indexInput = indexTarget;
            }
            return cost;
        }

        private void UpdateParameters()
        {
            var model = encoder.GetData();
            model.AddRange(decoder.GetData());
            model.AddRange(ReversEncoder.GetData());
            model.Add(Embedding);
            model.Add(WeightsOutput);
            model.Add(BiasOutput);
            solver.Optimize(model, learning_rate, RegularizationStrength, clipval);
        }

        private void Reset()
        {
            encoder.Reset();
            ReversEncoder.Reset();
            decoder.Reset();
        }
    }
}
