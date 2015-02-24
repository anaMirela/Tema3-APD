#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "mpi.h"
#define NUM_COLORS 256

//calculul modulului unui numar complex
double modul(double x_re, double x_im) {
	double rezultat;
	rezultat = sqrt(pow(x_re, 2) + pow(x_im, 2));
	return rezultat;
}

//calculul coordonatei reale a patratului unui numar complex
double pow2_real(double x_re, double x_im) {
	double rezultat;
	rezultat = pow(x_re, 2) - pow(x_im, 2);
	return rezultat;
}

//calculul coordonatei imaginare a patratului unui numar complex
double pow2_imaginar(double x_re, double x_im) {
	double rezultat;
	rezultat = 2 * x_re * x_im;
	return rezultat;
}

int main(int argc, char **argv) {

	int n, rank;
	MPI_Status stat;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n);

	FILE *f_in, *f_out;
	double x_min, x_max, y_min, y_max;
	double rezolutie, cx_julia, cy_julia, send_buffer_double[7];
	int MAX_STEPS, tip_multime, send_buffer_int[2];
	int i, j, p;

	//procesul master deschide si citeste datele din fisierul de intrare
	if (rank == 0) {
		f_in = fopen(argv[1], "r");
		fscanf(f_in, "%d", &tip_multime);
		fscanf(f_in, "%lf %lf %lf %lf", &x_min, &x_max, &y_min, &y_max);
		fscanf(f_in, "%lf", &rezolutie);
		fscanf(f_in, "%d", &MAX_STEPS);

		if (tip_multime == 1) {
			fscanf(f_in, "%lf %lf", &cx_julia, &cy_julia);
		}
		fclose(f_in);

		//buffer cu date de tip double ce trebuie trimise
		//catre restul proceselor
		send_buffer_double[0] = x_min;
		send_buffer_double[1] = x_max;
		send_buffer_double[2] = y_min;
		send_buffer_double[3] = y_max;
		send_buffer_double[4] = rezolutie;
		send_buffer_double[5] = cx_julia;
		send_buffer_double[6] = cy_julia;

		//buffer cu date de tip int ce trebuie trimise
		//catre restul proceselor
		send_buffer_int[0] = MAX_STEPS;
		send_buffer_int[1] = tip_multime;

	}

	//procesul master face broadcast datelor din fisier
	MPI_Bcast(send_buffer_double, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(send_buffer_int, 2, MPI_INT, 0, MPI_COMM_WORLD);

	//procesele isi salveaza valorile primite din buffer
	if (rank != 0) {

		x_min = send_buffer_double[0];
		x_max = send_buffer_double[1];
		y_min = send_buffer_double[2];
		y_max = send_buffer_double[3];
		rezolutie = send_buffer_double[4];
		cx_julia = send_buffer_double[5];
		cy_julia = send_buffer_double[6];

		MAX_STEPS = send_buffer_int[0];
		tip_multime = send_buffer_int[1];

	}

	//dimensiunile imaginii
	int height = floor((y_max - y_min) / rezolutie);
	int width = floor((x_max - x_min) / rezolutie);

	//intervalul pe care il prelucreaza un proces
	int width_proc = height / n;

	//vector cu valorile calculate de fiecare proces
	int v[height * width_proc];
	int step, k;

	//multimea Mandelbrot
	if (tip_multime == 0) {
		k = 0;
		double z_re, z_im, z_re_initial, cx, cy;
		for (i = 0; i < height; i++)
			for (j = 0; j < width_proc; j++) {
				z_re = 0.0;
				z_im = 0.0;
				step = 0;
				cx = x_min + j * rezolutie + rank * rezolutie * width_proc;
				cy = y_min + i * rezolutie;
				while (modul(z_re, z_im) < 2.0 && step < MAX_STEPS) {
					z_re_initial = z_re;
					z_re = pow2_real(z_re_initial, z_im) + cx;
					z_im = pow2_imaginar(z_re_initial, z_im) + cy;
					step++;
				}

				v[k] = step % NUM_COLORS;
				k++;
			}
	} else { // multimea Julia
		k = 0;
		for (i = 0; i < height; i++)
			for (j = 0; j < width_proc; j++) {

				double z_re = x_min + j * rezolutie
						+ rank * rezolutie * width_proc;
				double z_im = y_min + i * rezolutie;
				step = 0;

				while (modul(z_re, z_im) < 2.0 && step < MAX_STEPS) {
					double z_re_initial = z_re;
					z_re = pow2_real(z_re_initial, z_im) + cx_julia;
					z_im = pow2_imaginar(z_re_initial, z_im) + cy_julia;
					step++;
				}
				v[k] = step % NUM_COLORS;
				k++;
			}

	}
	//vector ce strange valorile calculate de fiecare proces
	int v_gather[height * width_proc * n];
	MPI_Gather(&v, height * width_proc, MPI_INT, &v_gather, height * width_proc,
			MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		//matricea imaginii finale
		int final_image[height][width];
		for (i = 0; i < height; i++)
			for (p = 0; p < n; p++)
				for (j = 0; j < width_proc; j++) {
					final_image[i][p * width_proc + j] = v_gather[p * height
							* width_proc + j + i * width_proc];
				}

		//scrierea in fisierul de iesire a imaginii
		f_out = fopen(argv[2], "w");
		fprintf(f_out, "P2\n");
		fprintf(f_out, "%d %d\n", width, height);
		fprintf(f_out, "255\n");

		for (i = height - 1; i >= 0; i--) {
			for (j = 0; j < width; j++)
				fprintf(f_out, "%d ", final_image[i][j]);
			fprintf(f_out, "\n");
		}

		fclose(f_out);

	}
	MPI_Finalize();

	return 0;
}

