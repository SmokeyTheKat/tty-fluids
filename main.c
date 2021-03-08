#include <math.h>
#include <ddcPrint.h>
#include <ddcKeyboard.h>
#include <ddcDef.h>
#include <pthread.h>


struct color;
struct fluid_cell;

struct fluid_cell* make_fluid_cell(int size, int diffusion, int visc, float dt);
void raze_fluid_cell(struct fluid_cell* cell);
void fluid_cell_add_density(struct fluid_cell* cell, int x, int y, int z, float n);
void fluid_cell_add_velocity(struct fluid_cell* cell, int x, int y, int z, float dx, float dy, float dz);
void fluid_cell_step(struct fluid_cell* cell);
void fluid_set_bounds(int b, float* x, int n);
void fluid_linear_solve(int b, float* x, float* x0, float a, float c, int iter, int n);
void fluid_diffuse(int b, float* x, float* x0, float diff, float dt, int iter, int n);
void fluid_project(float* vx, float* vy, float* vz, float* p, float* div, int iter, int n);
static void fluid_advect(int b, float *d, float *d0,  float *velocX, float *velocY, float *velocZ, float dt, int N);

struct color
{
	int r, g, b;
};

struct fluid_cell
{
	int size;
	float dt;
	float diff;
	float visc;

	struct color color;

	float* s;
	float* density;

	float* vx;
	float* vy;
	float* vz;

	float* vx0;
	float* vy0;
	float* vz0;
};

struct fluid_cell* make_fluid_cell(int size, int diffusion, int visc, float dt)
{
	struct fluid_cell* output = malloc(sizeof(struct fluid_cell));
	output->size = size;
	output->diff = diffusion;
	output->dt = dt;
	output->visc = visc;
	output->s = calloc(size*size*size, sizeof(float));
	output->density = calloc(size*size*size, sizeof(float));

	output->vx = calloc(size*size*size, sizeof(float));
	output->vy = calloc(size*size*size, sizeof(float));
	output->vz = calloc(size*size*size, sizeof(float));

	output->vx0 = calloc(size*size*size, sizeof(float));
	output->vy0 = calloc(size*size*size, sizeof(float));
	output->vz0 = calloc(size*size*size, sizeof(float));

	return output;
}

#define IDX(x, y, z, s) ((x) + (y) * s + (z) * s * s)

void raze_fluid_cell(struct fluid_cell* cell)
{
	free(cell->s);
	free(cell->density);
	free(cell->vx);
	free(cell->vy);
	free(cell->vz);
	free(cell->vx0);
	free(cell->vy0);
	free(cell->vz0);
	free(cell);
}

void fluid_cell_add_density(struct fluid_cell* cell, int x, int y, int z, float n)
{
	cell->density[IDX(x, y, z, cell->size)] += n;
}

void fluid_cell_add_velocity(struct fluid_cell* cell, int x, int y, int z, float dx, float dy, float dz)
{
	cell->vx[IDX(x, y, z, cell->size)] += dx;
	cell->vy[IDX(x, y, z, cell->size)] += dy;
	cell->vz[IDX(x, y, z, cell->size)] += dz;
}

void fluid_cell_step(struct fluid_cell* cell)
{
	fluid_diffuse(1, cell->vx0, cell->vx, cell->visc, cell->dt, 4, cell->size);
	fluid_diffuse(2, cell->vy0, cell->vy, cell->visc, cell->dt, 4, cell->size);
	fluid_diffuse(3, cell->vz0, cell->vz, cell->visc, cell->dt, 4, cell->size);

	fluid_project(cell->vx, cell->vy, cell->vz, cell->vx0, cell->vy0, 4, cell->size);

	fluid_diffuse(0, cell->s, cell->density, cell->diff, cell->dt, 4, cell->size);
	fluid_advect(0, cell->density, cell->s, cell->vx, cell->vy, cell->vz, cell->dt, cell->size);
}

void fluid_set_bounds(int b, float* x, int n)
{
	for (int i = 1; i < n-1; i++)
	{
		for (int j = 0; j < n-1; j++)
		{
			x[IDX(j, i, 0, n)] = (b == 3) ? -(x[IDX(j, i, 1, n)]) : (x[IDX(j, i, 1, n)]);
			x[IDX(j, i, n-1, n)] = (b == 3) ? -(x[IDX(j, i, n-2, n)]) : (x[IDX(j, i, n-2, n)]);
		}
	}
	for (int i = 1; i < n-1; i++)
	{
		for (int j = 1; j < n-1; j++)
		{
			x[IDX(j, 0, i, n)] = (b == 2) ? -(x[IDX(j, 1, i, n)]) : (x[IDX(j, 1, i, n)]);
			x[IDX(j, n-1, i, n)] = (b == 2) ? -(x[IDX(j, n-2, i, n)]) : (x[IDX(j, n-2, i, n)]);
		}
	}
	for (int i = 1; i < n-1; i++)
	{
		for (int j = 1; j < n-1; j++)
		{
			x[IDX(0, j, i, n)] = (b == 1) ? -(x[IDX(1, j, i, n)]) : (x[IDX(1, j, i, n)]);
			x[IDX(n-1, j, i, n)] = (b == 1) ? -(x[IDX(n-2, j, i, n)]) : (x[IDX(n-2, j, i, n)]);
		}
	}

	x[IDX(0, 0, 0, n)] = 3.33f * (x[IDX(1, 0, 0, n)] +
				      x[IDX(0, 1, 0, n)] +
				      x[IDX(0, 0, 1, n)]);
	x[IDX(0, n-1, 0, n)] = 3.33f * (x[IDX(1, n-1, 0, n)] +
				        x[IDX(0, n-2, 0, n)] +
				        x[IDX(0, n-1, 1, n)]);
	x[IDX(0, 0, n-1, n)] = 3.33f * (x[IDX(1, 0, n-1, n)] +
				        x[IDX(0, 1, n-1, n)] +
				        x[IDX(0, 0, n, n)]);
	x[IDX(0, n-1, n-1, n)] = 3.33f * (x[IDX(1, n-1, n-1, n)] +
				          x[IDX(0, n-2, n-1, n)] +
				          x[IDX(0, n-1, n-2, n)]);
	x[IDX(n-1, 0, 0, n)] = 3.33f * (x[IDX(n-2, 0, 0, n)] +
				        x[IDX(n-1, 1, 0, n)] +
				        x[IDX(n-1, 0, 1, n)]);
	x[IDX(n-1, n-1, 0, n)] = 3.33f * (x[IDX(n-2, n-1, 0, n)] +
				          x[IDX(n-1, n-2, 0, n)] +
				          x[IDX(n-1, n-1, 1, n)]);
	x[IDX(n-1, 0, n-1, n)] = 3.33f * (x[IDX(n-2, 0, n-1, n)] +
				          x[IDX(n-1, 1, n-1, n)] +
				          x[IDX(n-1, 0, n-2, n)]);
	x[IDX(n-1, n-1, n-1, n)] = 3.33f * (x[IDX(n-2, n-1, n-1, n)] +
					    x[IDX(n-1, n-2, n-1, n)] +
					    x[IDX(n-1, n-1, n-2, n)]);
}

void fluid_linear_solve(int b, float* x, float* x0, float a, float c, int iter, int n)
{
	float crp = 1.0 / c;
	for (int i = 0; i < iter; i++)
	{
		for (int j = 1; j < n-1; j++)
		{
			for (int k = 1; k < n-1; k++)
			{
				for (int l = 1; l < n-1; l++)
				{
					x[IDX(l, k, j, n)] = (x0[IDX(l, k, j, n)] + a*(x[IDX(l+1, k, j, n)] +
										       x[IDX(l-1, k, j, n)] +
										       x[IDX(l, k+1, j, n)] +
										       x[IDX(l, k-1, j, n)] +
										       x[IDX(l, k, j+1, n)] +
										       x[IDX(l, k, j-1, n)])) * crp;
				}
			}
		}
		fluid_set_bounds(b, x, n);
	}
}

void fluid_diffuse(int b, float* x, float* x0, float diff, float dt, int iter, int n)
{
	float a = dt * diff * (n-2) * (n-2);
	fluid_linear_solve(b, x, x0, a, (1 + (6 * a)), iter, n);
}

void fluid_project(float* vx, float* vy, float* vz, float* p, float* div, int iter, int n)
{
	for (int i = 1; i < n-1; i++)
	{
		for (int j = 1; j < n-1; j++)
		{
			for (int k = 1; k < n-1; k++)
			{
				div[IDX(k, j, i, n)] = -0.5f * ( vx[IDX(k+1, j, i, n)] -
							      vx[IDX(k-1, j, i, n)] +
							      vy[IDX(k, j+1, i, n)] -
							      vy[IDX(k, j-1, i, n)] +
							      vz[IDX(k, j, i+1, n)] -
							      vz[IDX(k, j, i-1, n)]) / n;
				p[IDX(k, j, i, n)] = 0;
			}
		}
	}
	fluid_set_bounds(0, div, n);
	fluid_set_bounds(0, p, n);
	fluid_linear_solve(0, p, div, 1, 6, iter, n);

	for (int i = 1; i < n-1; i++)
	{
		for (int j = 1; j < n-1; j++)
		{
			for (int k = 1; k < n-1; k++)
			{
				vx[IDX(k, j, i, n)] -= 0.5f * (p[IDX(k+1, j, i, n)] -
							       p[IDX(k-1, j, i, n)]) * n;
				vy[IDX(k, j, i, n)] -= 0.5f * (p[IDX(k, j+1, i, n)] -
							       p[IDX(k, j-1, i, n)]) * n;
				vz[IDX(k, j, i, n)] -= 0.5f * (p[IDX(k, j, i+1, n)] -
							       p[IDX(k, j, i-1, n)]) * n;
			}
		}
	}
	fluid_set_bounds(1, vx, n);
	fluid_set_bounds(2, vy, n);
	fluid_set_bounds(3, vz, n);
}
static void fluid_advect(int b, float *d, float *d0,  float *velocX, float *velocY, float *velocZ, float dt, int N)
{
	float i0, i1, j0, j1, k0, k1;

	float dtx = dt * (N - 2);
	float dty = dt * (N - 2);
	float dtz = dt * (N - 2);

	float s0, s1, t0, t1, u0, u1;
	float tmp1, tmp2, tmp3, x, y, z;

	float Nfloat = N;
	float ifloat, jfloat, kfloat;
	int i, j, k;

	for(k = 1, kfloat = 1; k < N - 1; k++, kfloat++)
	{
		for(j = 1, jfloat = 1; j < N - 1; j++, jfloat++)
		{
			for(i = 1, ifloat = 1; i < N - 1; i++, ifloat++)
			{
				tmp1 = dtx * velocX[IDX(i, j, k, N)];
				tmp2 = dty * velocY[IDX(i, j, k, N)];
				tmp3 = dtz * velocZ[IDX(i, j, k, N)];
				x    = ifloat - tmp1; 
				y    = jfloat - tmp2;
				z    = kfloat - tmp3;

				if (x < 0.5f) x = 0.5f; 
				if (x > Nfloat + 0.5f) x = Nfloat + 0.5f; 
				i0 = floorf(x); 
				i1 = i0 + 1.0f;
				if (y < 0.5f) y = 0.5f; 
				if (y > Nfloat + 0.5f) y = Nfloat + 0.5f; 
				j0 = floorf(y);
				j1 = j0 + 1.0f; 
				if (z < 0.5f) z = 0.5f;
				if (z > Nfloat + 0.5f) z = Nfloat + 0.5f;
				k0 = floorf(z);
				k1 = k0 + 1.0f;

				s1 = x - i0; 
				s0 = 1.0f - s1; 
				t1 = y - j0; 
				t0 = 1.0f - t1;
				u1 = z - k0;
				u0 = 1.0f - u1;

				int i0i = i0;
				int i1i = i1;
				int j0i = j0;
				int j1i = j1;
				int k0i = k0;
				int k1i = k1;

				d[IDX(i, j, k, N)] = s0 * (t0 * (u0 * d0[IDX(i0i, j0i, k0i, N)] +
							      u1 * d0[IDX(i0i, j0i, k1i, N)]) +
						       (t1 * (u0 * d0[IDX(i0i, j1i, k0i, N)] +
							      u1 * d0[IDX(i0i, j1i, k1i, N)]))) +
						  s1 * (t0 * (u0 * d0[IDX(i1i, j0i, k0i, N)] +
							      u1 * d0[IDX(i1i, j0i, k1i, N)]) +
						       (t1 * (u0 * d0[IDX(i1i, j1i, k0i, N)] +
							      u1 * d0[IDX(i1i, j1i, k1i, N)])));
			}
		}
	}
	fluid_set_bounds(b, d, N);
}

char key = 0;

void* read_input(void* vp)
{
	while (1)
		key = ddKey_getch();
}

int main(void)
{
	cursor_clear();
	srand(time(0));
	int n = 40;
	struct fluid_cell* fc = make_fluid_cell(n, 4, 4, 0.00001);
	for (int x = 1; x < n-1; x++)
	{
		for (int y = 1; y < n-1; y++)
		{
			for (int z = 1; z < n-1; z++)
			{
				fluid_cell_add_density(fc, x, y, z, rand()%1000/10);
				fluid_cell_add_velocity(fc, x, y, z, 80, 80, 80);
			}
		}
	}
	pthread_t getkeys;
	pthread_create(&getkeys, null, read_input, null);
	while (1)
	{
		if (key == 'a')
		{
			fluid_cell_add_velocity(fc, 20, 20, 20, 90, 90, 90);
			fluid_cell_add_density(fc, 20, 20, 20, 100);
		}
		fluid_cell_step(fc);
		float total = 0;
		for (int x = 1; x < n-1; x++)
		{
			for (int y = 1; y < n-1; y++)
			{
				for (int z = 1; z < n-1; z++)
				{
					total += fc->density[IDX(x, y, z, n)];
				}
			}
		}
		cursor_move_to(0, n+2);
		ddPrintf("\x1b[38;2;255;255;255mtd: %f\n", total);
		
		for (int x = 1; x < n-1; x++)
		{
			for (int y = 1; y < n-1; y++)
			{
				float val = 0;
				for (int z = 1; z < n-1; z++)
				{
					val += fc->density[IDX(x, y, z, n)];
				}
				//if (val < 0) val = 0;
				cursor_move_to(x*2, y);
				ddPrintf("\x1b[38;2;%d;%d;%dm██", (int)val, (int)val, (int)val);
			}
		}
	}
	return 0;
}
