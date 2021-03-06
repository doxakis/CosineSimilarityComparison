﻿using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace CosineSimilarityComparison.Versions
{
	public class VectorizedV2CosineSimilarityDoubleVersion
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

			int i = 0;
			if (Vector.IsHardwareAccelerated)
			{
				int s = Vector<double>.Count;
				int n = attributesOne.Length / s * s;
				Vector<double> dotProductTemp = new Vector<double>();
				Vector<double> magnitudeOneTemp = new Vector<double>();
				Vector<double> magnitudeTwoTemp = new Vector<double>();
				
				for (i = 0; i < n; i += s)
				{
					var one = new Vector<double>(attributesOne, i);
					var two = new Vector<double>(attributesTwo, i);

					dotProductTemp = dotProductTemp + one * two;
					magnitudeOneTemp = magnitudeOneTemp + one * one;
					magnitudeTwoTemp = magnitudeTwoTemp + two * two;
				}

				var pool = ArrayPool<double>.Shared;
				double[] result = pool.Rent(s);

				dotProductTemp.CopyTo(result);
				dotProduct = result.Sum();

				magnitudeOneTemp.CopyTo(result);
				magnitudeOne = result.Sum();

				magnitudeTwoTemp.CopyTo(result);
				magnitudeTwo = result.Sum();

				pool.Return(result);
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
