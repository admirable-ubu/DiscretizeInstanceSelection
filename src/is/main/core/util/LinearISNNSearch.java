/*
 * This file is part of Instance Selection Library.
 * 
 * Instance Selection Library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Instance Selection Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Instance Selection Library.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * LinearISNNSearch.java
 * Copyright (C) 2010 Universidad de Burgos
 */

package main.core.util;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * <b>Descripción</b><br>
 * Algoritmo de cálculo de vecinos cercanos.
 * <p>
 * <b>Detalles</b><br>
 * Se asegura de que la instancia de la que se desea obtener sus vecinos no sea devuelta.  
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Cálculo de vecinos cercanos.
 * </p>
 * 
 * @author Álvar Arnáiz González
 * @version 1.1
 */
public class LinearISNNSearch extends LinearNNSearch {

	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = 2038643255873524858L;

	/**
	 * Constructor por defecto.
	 * Es necesario asingar las instancias <code>setInstances(inst)</code> antes de poder utilizarlo.
	 */
	public LinearISNNSearch() {
		super();
		
		// El algoritmo de cálculo de distancias NO debe normalizar las instancias.
		((NormalizableDistance)m_DistanceFunction).setDontNormalize(true);
	} // LinearISNNSearch

	/**
	 * Constructor al que se le asigna un conjunto de instancias para utilizar.
	 * 
	 * @param insts Instancias a utilizar.
	 */
	public LinearISNNSearch(Instances insts) {
		super(insts);
		
		// El algoritmo de cálculo de distancias NO debe normalizar las instancias.
		((NormalizableDistance)m_DistanceFunction).setDontNormalize(true);
	} // LinearISNNSearch
	
	/**
	 * Devuelve el vecino más próximo.
	 * Si no hay ningún vecino cercano devolverá la misma instancia.<br>
	 * Si hubiese varios vecinos a la misma distancia devolverá uno de ellos, no se asegura el orden.
	 * 
	 * @param target Instancia de la cual se desea obtener el vecino más próximo.
	 * @return Vecino más próximo o la propia instancia si no tiene vecinos.
	 * @throws Exception Si se produce algún error en el cálculo de instancias.
	 */
	public Instance nearestNeighbour(Instance target) throws Exception {
		Instances nearestNeighbours = kNearestNeighbours(target, 1);
		
		// Si no tiene vecinos próximos devolver target.
		if (nearestNeighbours.numInstances() == 0)
			return target;
		
		return nearestNeighbours.firstInstance();
	} // nearestNeighbour
	
	/**
	 * Devuelve las "k" instancias más cercanas a la instancia dada. 
	 * 
	 * @param target Instancia de la que se desean obtener sus k vecinos cercanos.
	 * @param kNN Número de vecinos cercanos a obtener.
	 * @return Los k vecinos cercanos.
	 * @throws Exception Si se produce algún error en el cálculo de instancias.
	 */
	public Instances kNearestNeighbours(Instance target, int kNN) throws Exception {
		MyHeap heap = new MyHeap(kNN);
		double distance;
		int firstkNN = 0;
		
		// Recorrer todas las instancias del conjunto de entrenamiento de la clase.
		for (int i = 0; i < m_Instances.numInstances(); i++) {
			// Si la instancia del conjunto de entrenamiento es igual a la instancia objetivo no tenerla en
			// cuenta.
			if (InstanceIS.equals(target, m_Instances.instance(i)))
				continue;
			
			if (firstkNN < kNN) {
				distance = m_DistanceFunction.distance(target, m_Instances.instance(i),
				                                        Double.POSITIVE_INFINITY);
				
				heap.put(i, distance);
				firstkNN++;
			} else {
				MyHeapElement temp = heap.peek();
				distance = m_DistanceFunction.distance(target, m_Instances.instance(i), temp.distance);
				if (distance < temp.distance) {
					heap.putBySubstitute(i, distance);
				} else if (distance == temp.distance) {
					heap.putKthNearest(i, distance);
				}
			}
		}

		Instances neighbours = new Instances(m_Instances, (heap.size() + heap.noOfKthNearest()));
		m_Distances = new double[heap.size() + heap.noOfKthNearest()];
		int[] indices = new int[heap.size() + heap.noOfKthNearest()];
		int i = 1;
		MyHeapElement h;
		
		while (heap.noOfKthNearest() > 0) {
			h = heap.getKthNearest();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = h.distance;
			i++;
		}
		
		while (heap.size() > 0) {
			h = heap.get();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = h.distance;
			i++;
		}

		m_DistanceFunction.postProcessDistances(m_Distances);

		for (int k = 0; k < indices.length; k++)
			neighbours.add(m_Instances.instance(indices[k]));
		
		return neighbours;
	} // kNearestNeighbours
	
} // LinearISNNSearch
