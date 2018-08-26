using CosineSimilarityComparison.Versions;
using System;
using System.Diagnostics;

namespace CosineSimilarityComparison
{
    class Program
    {
        static void Main(string[] args)
		{
			Console.WriteLine("Integer vs float:");

			{
				int[][] integerDataSet = GenerateIntegerDataSet(200, 100000);
				float[][] floatDataSet = GenerateFloatDataSet(200, 100000);

				{
					var watch = Stopwatch.StartNew();
					SimpleCosineSimilarityIntegerVersion.ComputeDistances(integerDataSet);
					Console.WriteLine("Compute with integer, result with double: " + watch.ElapsedMilliseconds + " ms");
				}

				{
					var watch = Stopwatch.StartNew();
					SimpleCosineSimilarityIntegerVersionResultInFloat.ComputeDistances(integerDataSet);
					Console.WriteLine("Compute with integer, result with float: " + watch.ElapsedMilliseconds + " ms");
				}

				{
					var watch = Stopwatch.StartNew();
					SimpleCosineSimilarityFloatVersion.ComputeDistances(floatDataSet);
					Console.WriteLine("Compute with float, result with float: " + watch.ElapsedMilliseconds + " ms");
				}
			}

			Console.WriteLine("\nFloat versions:\n");

			// JIT compilation.
			{
				try
				{
					var watch = Stopwatch.StartNew();
					float[][] smallDataSet = GenerateFloatDataSet(1, 1);
					GpuCosineSimilarityFloatVersion.ComputeDistances(smallDataSet);
					Console.WriteLine("Gpu (JIT compilation): " + watch.ElapsedMilliseconds + " ms");
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu (JIT compilation): Exception: " + ex.Message);
				}
			}

			RunComparisonFloatVersions(200, 100000);
			RunComparisonFloatVersions(2000, 5000);
			RunComparisonFloatVersions(5000, 25);

			Console.WriteLine("\nInteger versions:\n");

			// JIT compilation.
			{
				try
				{
					var watch = Stopwatch.StartNew();
					int[][] smallDataSet = GenerateIntegerDataSet(1, 1);
					GpuCosineSimilarityIntegerVersion.ComputeDistances(smallDataSet);
					Console.WriteLine("Gpu (JIT compilation): " + watch.ElapsedMilliseconds + " ms");
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu (JIT compilation): Exception: " + ex.Message);
				}
			}
			
			RunComparisonIntegerVersions(200, 100000);
			RunComparisonIntegerVersions(2000, 5000);
			RunComparisonIntegerVersions(5000, 25);

			// Demonstrate GPU communication cost.
			RunComparisonIntegerVersions(1, 1);

			Console.WriteLine("\nDouble versions:\n");

			// JIT compilation.
			{
				try
				{
					var watch = Stopwatch.StartNew();
					double[][] smallDataSet = GenerateDoubleDataSet(1, 1);
					GpuCosineSimilarityDoubleVersion.ComputeDistances(smallDataSet);
					Console.WriteLine("Gpu (JIT compilation): " + watch.ElapsedMilliseconds + " ms");
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu (JIT compilation): Exception: " + ex.Message);
				}
			}

			RunComparisonDoubleVersions(2000, 5000);
			RunComparisonDoubleVersions(5000, 25);

			// Demonstrate the GPU limit.
			{
				try
				{
					int numElement = 10000;
					int numDimension = 1000;
					Console.WriteLine("\nDataset: "+ numElement + "x" + numDimension);
					double[][] dataSet = GenerateDoubleDataSet(numElement, numDimension);
					var result = GpuCosineSimilarityDoubleVersion.ComputeDistances(dataSet);
					Console.WriteLine("Gpu:                    No error");
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu:                    Exception: " + ex.Message);
				}
			}

			Console.WriteLine("\nPress enter to continue...");
			Console.ReadLine();
		}

		private static void RunComparisonFloatVersions(int numElement, int numDimension)
		{
			float[][] dataSet = GenerateFloatDataSet(numElement, numDimension);
			Console.WriteLine("\nDataset: " + numElement + "x" + numDimension);
			float[][] distances;

			{
				var watch = Stopwatch.StartNew();
				distances = SimpleCosineSimilarityFloatVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("Simple 1 thread:        " + watch.ElapsedMilliseconds + " ms");
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityFloatVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("Simple 2 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityFloatVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("Simple 4 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityFloatVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("Simple 8 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				try
				{
					var watch = Stopwatch.StartNew();
					var result = GpuCosineSimilarityFloatVersion.ComputeDistances(dataSet);
					Console.WriteLine("Gpu:                    " + watch.ElapsedMilliseconds + " ms");
					ValidateSameResult(distances, result);
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu:                    Exception: " + ex.Message);
				}
			}

			// GPU with cached kernel.
			Console.WriteLine("GpuCosineSimilarityFloatVersionCacheKernel:");
			{
				try
				{
					var instance = new GpuCosineSimilarityFloatVersionCacheKernel();
					
					{
						var watch = Stopwatch.StartNew();
						instance.Init();
						Console.WriteLine("    (init):             " + watch.ElapsedMilliseconds + " ms");
					}

					// Test GPU latency and find out if GPU duration is stable within multiple call.
					long min = long.MaxValue;
					long max = long.MinValue;

					for (int i = 0; i < 20; i++)
					{
						var watch = Stopwatch.StartNew();
						var result = instance.ComputeDistances(dataSet);
						long elapsedMilliseconds = watch.ElapsedMilliseconds;
						if (elapsedMilliseconds < min)
							min = elapsedMilliseconds;
						if (elapsedMilliseconds > max)
							max = elapsedMilliseconds;
						Console.WriteLine("    (ComputeDistances): " + elapsedMilliseconds + " ms");
						ValidateSameResult(distances, result);
					}
					Console.WriteLine("    min: " + min + " ms");
					Console.WriteLine("    max: " + max + " ms");

					{
						var watch = Stopwatch.StartNew();
						instance.Dispose();
						Console.WriteLine("    (dispose):          " + watch.ElapsedMilliseconds + " ms");
					}

				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu:                    Exception: " + ex.Message);
				}
			}
		}
		
		private static void RunComparisonIntegerVersions(int numElement, int numDimension)
		{
			int[][] dataSet = GenerateIntegerDataSet(numElement, numDimension);

			Console.WriteLine("\nDataset: " + numElement + "x" + numDimension);

			double[][] distances;
			{
				var watch = Stopwatch.StartNew();
				distances = SimpleCosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("Simple 1 thread:        " + watch.ElapsedMilliseconds + " ms");
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("Simple 2 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("Simple 4 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("Simple 8 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("SimpleV2 1 thread:      " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("SimpleV2 2 threads:     " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("SimpleV2 4 threads:     " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("SimpleV2 8 threads:     " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("VectorizedV1 1 thread:  " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("VectorizedV1 2 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("VectorizedV1 4 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("VectorizedV1 8 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("VectorizedV2 1 thread:  " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("VectorizedV2 2 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("VectorizedV2 4 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityIntegerVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("VectorizedV2 8 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				try
				{
					var watch = Stopwatch.StartNew();
					var result = GpuCosineSimilarityIntegerVersion.ComputeDistances(dataSet);
					Console.WriteLine("Gpu:                    " + watch.ElapsedMilliseconds + " ms");
					ValidateSameResult(distances, result);
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu:                    Exception: " + ex.Message);
				}
			}

			// GPU with cached kernel.
			Console.WriteLine("GpuCosineSimilarityIntegerVersionCacheKernel:");
			{
				try
				{
					var instance = new GpuCosineSimilarityIntegerVersionCacheKernel();

					{
						var watch = Stopwatch.StartNew();
						instance.Init();
						Console.WriteLine("    (init):             " + watch.ElapsedMilliseconds + " ms");
					}

					// Test GPU latency and find out if GPU duration is stable within multiple call.
					long min = long.MaxValue;
					long max = long.MinValue;

					for (int i = 0; i < 20; i++)
					{
						var watch = Stopwatch.StartNew();
						var result = instance.ComputeDistances(dataSet);
						long elapsedMilliseconds = watch.ElapsedMilliseconds;
						if (elapsedMilliseconds < min)
							min = elapsedMilliseconds;
						if (elapsedMilliseconds > max)
							max = elapsedMilliseconds;
						Console.WriteLine("    (ComputeDistances): " + elapsedMilliseconds + " ms");
						ValidateSameResult(distances, result);
					}
					Console.WriteLine("    min: " + min + " ms");
					Console.WriteLine("    max: " + max + " ms");

					{
						var watch = Stopwatch.StartNew();
						instance.Dispose();
						Console.WriteLine("    (dispose):          " + watch.ElapsedMilliseconds + " ms");
					}

				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu:                    Exception: " + ex.Message);
				}
			}

		}

		private static void RunComparisonDoubleVersions(int numElement, int numDimension)
		{
			double[][] dataSet = GenerateDoubleDataSet(numElement, numDimension);

			Console.WriteLine("\nDataset: " + numElement + "x" + numDimension);

			double[][] distances;
			{
				var watch = Stopwatch.StartNew();
				distances = SimpleCosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("Simple 1 thread:        " + watch.ElapsedMilliseconds + " ms");
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("Simple 2 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("Simple 4 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = SimpleCosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("Simple 8 threads:       " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("VectorizedV1 1 thread:  " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("VectorizedV1 2 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("VectorizedV1 4 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV1CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("VectorizedV1 8 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: false);
				Console.WriteLine("VectorizedV2 1 thread:  " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 2);
				Console.WriteLine("VectorizedV2 2 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 4);
				Console.WriteLine("VectorizedV2 4 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				var watch = Stopwatch.StartNew();
				var result = VectorizedV2CosineSimilarityDoubleVersion.ComputeDistances(dataSet, useMultipleThread: true, maxDegreeOfParallelism: 8);
				Console.WriteLine("VectorizedV2 8 threads: " + watch.ElapsedMilliseconds + " ms");
				ValidateSameResult(distances, result);
			}

			{
				try
				{
					var watch = Stopwatch.StartNew();
					var result = GpuCosineSimilarityDoubleVersion.ComputeDistances(dataSet);
					Console.WriteLine("Gpu:                    " + watch.ElapsedMilliseconds + " ms");
					ValidateSameResult(distances, result);
				}
				catch (Exception ex)
				{
					Console.WriteLine("Gpu:                    Exception: " + ex.Message);
				}
			}
		}

		private static void ValidateSameResult(double[][] distances, double[][] result)
		{
			double maxDiff = 0;
			for (int i = 0; i < distances.Length; i++)
			{
				for (int j = 0; j < distances[i].Length; j++)
				{
					double diff = Math.Abs(distances[i][j] - result[i][j]);
					if (diff > maxDiff)
					{
						maxDiff = diff;
					}
				}
			}
			if (maxDiff > 0.01 /* error margin */)
			{
				Console.WriteLine("Calculation error. Diff: " + maxDiff);
			}
		}

		private static void ValidateSameResult(float[][] distances, float[][] result)
		{
			double maxDiff = 0;
			for (int i = 0; i < distances.Length; i++)
			{
				for (int j = 0; j < distances[i].Length; j++)
				{
					double diff = Math.Abs(distances[i][j] - result[i][j]);
					if (diff > maxDiff)
					{
						maxDiff = diff;
					}
				}
			}
			if (maxDiff > 0.01 /* error margin */)
			{
				Console.WriteLine("Calculation error. Diff: " + maxDiff);
			}
		}

		private static int[][] GenerateIntegerDataSet(int numElement, int numDimension)
		{
			int[][] dataSet = new int[numElement][];
			Random random = new Random();
			for (int i = 0; i < dataSet.Length; i++)
			{
				dataSet[i] = new int[numDimension];
				for (int j = 0; j < dataSet[i].Length; j++)
				{
					dataSet[i][j] = random.Next(0, 4);
				}
			}
			return dataSet;
		}

		private static float[][] GenerateFloatDataSet(int numElement, int numDimension)
		{
			float[][] dataSet = new float[numElement][];
			Random random = new Random();
			for (int i = 0; i < dataSet.Length; i++)
			{
				dataSet[i] = new float[numDimension];
				for (int j = 0; j < dataSet[i].Length; j++)
				{
					dataSet[i][j] = random.Next(0, 4);
				}
			}
			return dataSet;
		}

		private static double[][] GenerateDoubleDataSet(int numElement, int numDimension)
		{
			double[][] dataSet = new double[numElement][];
			Random random = new Random();
			for (int i = 0; i < dataSet.Length; i++)
			{
				dataSet[i] = new double[numDimension];
				for (int j = 0; j < dataSet[i].Length; j++)
				{
					dataSet[i][j] = random.Next(0, 4);
				}
			}
			return dataSet;
		}
	}
}
