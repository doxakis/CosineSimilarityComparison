using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CosineSimilarityComparison.Versions
{
	/// <summary>
	/// From SimpleCosineSimilarityIntegerVersion, this version precalculate the magnitude.
	/// </summary>
	public class SimpleV2CosineSimilarityIntegerVersion
	{
		public static double[][] ComputeDistances(int[][] dataSet, bool useMultipleThread = false, int maxDegreeOfParallelism = 0)
		{
			int numPoints = dataSet.Length;
			double[][] distances = new double[numPoints][];
			for (int i = 0; i < distances.Length; i++)
			{
				distances[i] = new double[numPoints];
			}

			if (useMultipleThread)
			{
				int size = numPoints * numPoints;

				if (maxDegreeOfParallelism == 0)
				{
					// Not specified. Use all threads.
					maxDegreeOfParallelism = Environment.ProcessorCount;
				}
				var option = new ParallelOptions
				{
					MaxDegreeOfParallelism = Math.Max(1, maxDegreeOfParallelism)
				};

				double[] dataSetMagnitude = new double[dataSet.Length];
				Parallel.For(0, dataSet.Length, option, index =>
				{
					int i = index;
					double magnitude = 0;
					for (int j = 0; j < dataSet[i].Length; j++)
					{
						magnitude += dataSet[i][j] * dataSet[i][j];
					}
					dataSetMagnitude[i] = magnitude;
				});

				Parallel.For(0, size, option, index =>
				{
					int i = index % numPoints;
					int j = index / numPoints;
					if (i < j)
					{
						double distance = ComputeDistance(
								dataSet[i],
								dataSet[j],
							dataSetMagnitude[i],
							dataSetMagnitude[j]);
						distances[i][j] = distance;
						distances[j][i] = distance;
					}
				});
			}
			else
			{

				double[] dataSetMagnitude = new double[dataSet.Length];
				for (int i = 0; i < dataSet.Length; i++)
				{
					double magnitude = 0;
					for (int j = 0; j < dataSet[i].Length; j++)
					{
						magnitude += dataSet[i][j] * dataSet[i][j];
					}
					dataSetMagnitude[i] = magnitude;
				}

				for (int i = 0; i < numPoints; i++)
				{
					for (int j = 0; j < i; j++)
					{
						double distance = ComputeDistance(
							dataSet[i],
							dataSet[j],
							dataSetMagnitude[i],
							dataSetMagnitude[j]);
						distances[i][j] = distance;
						distances[j][i] = distance;
					}
				}
			}
			return distances;
		}

		public static double ComputeDistance(
			int[] attributesOne,
			int[] attributesTwo,
			double magnitudeOne,
			double magnitudeTwo)
		{
			double dotProduct = 0;

			for (int i = 0; i < attributesOne.Length && i < attributesTwo.Length; i++)
			{
				dotProduct += (attributesOne[i] * attributesTwo[i]);
			}
			return Math.Max(0, 1 - (dotProduct / Math.Sqrt(magnitudeOne * magnitudeTwo)));
		}
	}
}
