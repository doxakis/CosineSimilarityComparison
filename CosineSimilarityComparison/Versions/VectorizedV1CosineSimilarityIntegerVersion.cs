using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace CosineSimilarityComparison.Versions
{
	public class VectorizedV1CosineSimilarityIntegerVersion
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

		public static double ComputeDistance(int[] attributesOne, int[] attributesTwo)
		{
			double dotProduct = 0;
			double magnitudeOne = 0;
			double magnitudeTwo = 0;

			int i = 0;
			if (Vector.IsHardwareAccelerated)
			{
				int s = Vector<int>.Count;
				int n = attributesOne.Length / s * s;
				for (i = 0; i < n; i += s)
				{
					var one = new Vector<int>(attributesOne, i);
					var two = new Vector<int>(attributesTwo, i);
					dotProduct += Vector.Dot(one, two);
					magnitudeOne += Vector.Dot(one, one);
					magnitudeTwo += Vector.Dot(two, two);
				}
			}

			for (; i < attributesOne.Length; i++)
			{
				dotProduct += (attributesOne[i] * attributesTwo[i]);
				magnitudeOne += (attributesOne[i] * attributesOne[i]);
				magnitudeTwo += (attributesTwo[i] * attributesTwo[i]);
			}

			return Math.Max(0, 1 - (dotProduct / Math.Sqrt(magnitudeOne * magnitudeTwo)));
		}
	}
}
