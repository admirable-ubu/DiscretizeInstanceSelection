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
 * NotEnoughInstancesException.java
 * Copyright (C) 2010 Universidad de Burgos
 */

package main.core.exception;

/**
 * <b>Descripción</b><br>
 * Excepción que indica que no hay suficientes instancias.
 * <p>
 * <b>Detalles</b><br>
 * Esta excepción es lanzada cuando no existen suficientes instancias en el dataset.
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Indicar que no existen suficientes instancias en el dataset.
 * </p>
 * 
 * @author Álvar Arnáiz González
 * @version 1.1
 */
public class NotEnoughInstancesException extends Exception {

	/**
	 * Version UID.
	 */
	private static final long serialVersionUID = 5605896145908871507L;
	
	/**
	 * No existen suficientes instancias.
	 */
	public static final String MESSAGE = "No existen suficientes instancias";

	/**
	 * Constructor de la excepción.
	 * 
	 * @param arg0 Mensaje de la excepción.
	 */
	public NotEnoughInstancesException(String arg0) {
		super(arg0);
	} // NotEnoughInstancesException

} // NotEnoughInstancesException
