using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace CosineSimilarityComparison.Versions
{
    public class SimpleCosineSimilarityDoubleVersion
    {
		public static double[][] ComputeDistances(double[][] dataSet, bool useMultipleThread = false, int maxDegreeOfParallelism = 0)
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

				Parallel.For(0, size, option, index =>
				{
					int i = index % numPoints;
					int j = index / numPoints;
					if (i < j)
					{
						double distance = ComputeDistance(
								dataSet[i],
								dataSet[j]);
						distances[i][j] = distance;
						distances[j][i] = distance;
					}
				});
			}
			else
			{
				for (int i = 0; i < numPoints; i++)
				{
					for (int j = 0; j < i; j++)
					{
						double distance = ComputeDistance(
							dataSet[i],
							dataSet[j]);
						distances[i][j] = distance;
						distances[j][i] = distance;
					}
				}
			}
			return distances;
		}

		public static double ComputeDistance(double[] attributesOne, double[] attributesTwo)
		{
			double dotProduct = 0;
			double magnitudeOne = 0;
			double magnitudeTwo = 0;

			for (int i = 0; i < attributesOne.Length && i < attributesTwo.Length; i++)
			{
				dotProduct += (attributesOne[i] * attributesTwo[i]);
				magnitudeOne += (attributesOne[i] * attributesOne[i]);
				magnitudeTwo += (attributesTwo[i] * attributesTwo[i]);
			}
			return Math.Max(0, 1 - (dotProduct / Math.Sqrt(magnitudeOne * magnitudeTwo)));
		}
	}
}
