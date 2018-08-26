using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

namespace CosineSimilarityComparison.Versions
{
    public class SimpleCosineSimilarityFloatVersion
    {
		public static float[][] ComputeDistances(float[][] dataSet, bool useMultipleThread = false, int maxDegreeOfParallelism = 0)
		{
			int numPoints = dataSet.Length;
			float[][] distances = new float[numPoints][];
			for (int i = 0; i < distances.Length; i++)
			{
				distances[i] = new float[numPoints];
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
						float distance = ComputeDistance(
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
						float distance = ComputeDistance(
							dataSet[i],
							dataSet[j]);
						distances[i][j] = distance;
						distances[j][i] = distance;
					}
				}
			}
			return distances;
		}

		public static float ComputeDistance(float[] attributesOne, float[] attributesTwo)
		{
			float dotProduct = 0;
			float magnitudeOne = 0;
			float magnitudeTwo = 0;

			for (int i = 0; i < attributesOne.Length && i < attributesTwo.Length; i++)
			{
				dotProduct += (attributesOne[i] * attributesTwo[i]);
				magnitudeOne += (attributesOne[i] * attributesOne[i]);
				magnitudeTwo += (attributesTwo[i] * attributesTwo[i]);
			}

			return (float)Math.Max(0, 1 - (dotProduct / Math.Sqrt(magnitudeOne * magnitudeTwo)));
		}
	}
}
