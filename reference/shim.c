/* pachi-numba (GPL-2.0-only), October 2026: a C interface to Pachi for the tests; see NOTICE. */
/* A small C interface to Pachi for ctypes (see __init__.py).
 *
 * A game holds a Pachi board and a UCT engine. Points are numbered row by
 * row from the top-left corner (as in go_env); -1 is a pass, -2 a
 * resignation. Colors are Pachi's: 1 black, 2 white. The caller makes the
 * moves alternate: pachi_play() for the moves Pachi did not choose,
 * pachi_genmove() for Pachi's own moves. Pachi stops the process on
 * non-alternating moves, so __init__.py enforces the order.
 */

#include <assert.h>  /* mq.h uses assert without including it */
#include <stdio.h>
#include <stdlib.h>

#include "board.h"
#include "engine.h"
#include "move.h"
#include "mq.h"
#include "random.h"
#include "timeinfo.h"
#include "uct/uct.h"

struct pachi_game {
	struct board *board;
	struct engine *engine;
};

static coord_t
to_coord(struct board *b, int point)
{
	if (point == -1)
		return pass;
	int size = board_size(b) - 2;
	return coord_xy(b, point % size + 1, size - point / size);
}

static int
to_point(struct board *b, coord_t c)
{
	if (is_pass(c))
		return -1;
	if (is_resign(c))
		return -2;
	int size = board_size(b) - 2;
	return (size - coord_y(c, b)) * size + coord_x(c, b) - 1;
}

void *
pachi_new(int size, double komi, int threads)
{
	struct pachi_game *game = malloc(sizeof(*game));
	game->board = board_init(NULL);
	board_resize(game->board, size);
	board_clear(game->board);
	game->board->komi = komi;
	board_set_rules(game->board, "chinese");
	char options[64];
	/* Pondering would keep searching between the moves. */
	snprintf(options, sizeof(options), "threads=%d,pondering=0", threads);
	game->engine = engine_uct_init(options, game->board);
	return game;
}

void
pachi_free(void *handle)
{
	struct pachi_game *game = handle;
	struct engine *e = game->engine;
	if (e->stop)
		e->stop(e);
	if (e->done)
		e->done(e);
	free(e->data);
	free(e);
	board_done(game->board);
	free(game);
}

void
pachi_seed(unsigned long seed)
{
	fast_srandom(seed);
}

/* Play a move not chosen by Pachi; returns 0 if it is illegal. */
int
pachi_play(void *handle, int point, int color)
{
	struct pachi_game *game = handle;
	struct move m = { to_coord(game->board, point), color };
	struct board *trial = malloc(sizeof(*trial));
	board_copy(trial, game->board);
	int is_legal = board_play(trial, &m) >= 0;
	board_done(trial);
	if (!is_legal)
		return 0;
	if (game->engine->notify_play)
		game->engine->notify_play(game->engine, game->board, &m, NULL);
	board_play(game->board, &m);
	return 1;
}

/* Let Pachi choose and play a move with a number of random games. */
int
pachi_genmove(void *handle, int color, int playouts)
{
	struct pachi_game *game = handle;
	struct time_info ti = { .period = TT_NULL };
	char games[32];
	snprintf(games, sizeof(games), "=%d", playouts);
	time_parse(&ti, games);
	coord_t *c = game->engine->genmove(game->engine, game->board, &ti, color, false);
	coord_t coord = *c;
	coord_done(c);
	if (!is_resign(coord)) {
		struct move m = { coord, color };
		board_play(game->board, &m);
	}
	return to_point(game->board, coord);
}

/* Black's area minus white's area minus komi; Pachi's engine decides
 * which groups are dead and removes them first (as GTP final_score). */
double
pachi_black_margin(void *handle)
{
	struct pachi_game *game = handle;
	struct move_queue mq = { .moves = 0 };
	if (game->engine->dead_group_list)
		game->engine->dead_group_list(game->engine, game->board, &mq);
	return -board_official_score(game->board, &mq);
}

/* 0 empty, 1 black, 2 white. */
int
pachi_stone(void *handle, int point)
{
	struct pachi_game *game = handle;
	return board_at(game->board, to_coord(game->board, point));
}


/* ---- Exports for the tests of pachi-numba: the C original of every
 * translated function, called from Python through ctypes. ---- */

#include <string.h>
#include "stone.h"

unsigned int ref_fast_random(unsigned int max) { return fast_random(max); }
unsigned int ref_fast_irandom(unsigned int max) { return fast_irandom(max); }
float ref_fast_frandom(void) { return fast_frandom(); }
unsigned long ref_fast_getseed(void) { return fast_getseed(); }
int ref_stone_other(int s) { return stone_other((enum stone)s); }

int
ref_str2coord(const char *text, int size)
{
	coord_t *c = str2coord((char *)text, size);
	int value = *c;
	coord_done(c);
	return value;
}

/* coord2bstr on a real board of the given side (without margin). */
void
ref_coord2str(int c, int side, char *out, int out_size)
{
	struct board *b = board_init(NULL);
	board_resize(b, side);
	board_clear(b);
	char buf[8];
	strncpy(out, coord2bstr(buf, c, b), out_size - 1);
	out[out_size - 1] = 0;
	board_done(b);
}

/* ---- Boards for the tests of board.py. Coordinates are Pachi's own. */

void *
ref_board_new(int side)
{
	struct board *b = board_init(NULL);
	board_resize(b, side);
	board_clear(b);
	return b;
}

void ref_board_free(void *b) { board_done(b); }
void ref_board_set_komi(void *b, float komi) { ((struct board *)b)->komi = komi; }
int ref_board_set_rules(void *b, const char *name) { return board_set_rules(b, (char *)name); }
void ref_board_handicap(void *b, int stones) { board_handicap(b, stones, NULL); }

void *
ref_board_clone(void *from)
{
	return board_copy(malloc(sizeof(struct board)), from);
}

int
ref_board_play(void *b, int coord, int color)
{
	struct move m = { coord, color };
	return board_play(b, &m);
}

int
ref_board_play_random(void *b, int color)
{
	coord_t coord;
	board_play_random(b, color, &coord, NULL, NULL);
	return coord;
}

int ref_board_undo(void *b) { return board_undo(b); }
float ref_board_fast_score(void *b) { return board_fast_score(b); }

float
ref_board_official_score(void *b, const int *dead, int num_dead)
{
	struct move_queue q = { .moves = 0 };
	for (int i = 0; i < num_dead; i++)
		mq_add(&q, dead[i], 0);
	return board_official_score(b, &q);
}

/* The inline checks of board.h, one point at a time. */
int
ref_board_check(void *board, int which, int coord, int color)
{
	struct board *b = board;
	switch (which) {
	case 0: return board_is_valid_play(b, color, coord);
	case 1: return board_is_valid_play_no_suicide(b, color, coord);
	case 2: return board_safe_to_play(b, coord, color);
	case 3: return board_get_atari_neighbor(b, coord, color);
	case 4: return board_is_one_point_eye(b, coord, color);
	case 5: return board_is_false_eyelike(b, coord, color);
	case 6: return board_get_one_point_eye(b, coord);
	case 7: return board_coord_in_symmetry(b, coord);
	case 8: return group_at(b, coord) ? group_stone_count(b, group_at(b, coord), color) : -1;
	case 9: return board_is_eyelike(b, coord, color);
	case 10: return immediate_liberty_count(b, coord);
	}
	return -100;
}

/* Copy every field of the board out, in the layout of board.py. */
void
ref_board_dump(void *board, long *ints, long *captures, float *komi, long *nei8, long *dnei,
	       signed char *stones, int *g, int *p, signed char *n, unsigned long *h,
	       unsigned int *pat3, unsigned char *coord, int *gi_lib, int *gi_libs,
	       int *f, int *c, unsigned long *history_hash, unsigned long *hash, unsigned long *qhash)
{
	struct board *b = board;
	long scalars[] = {
		b->size, b->size2, b->bits2, b->handicap, b->rules, b->moves, b->superko_violation,
		b->flen, b->clen, b->last_ko_age,
		b->last_move.coord, b->last_move.color, b->last_move2.coord, b->last_move2.color,
		b->last_move3.coord, b->last_move3.color, b->last_move4.coord, b->last_move4.color,
		b->last_ko.coord, b->last_ko.color, b->ko.coord, b->ko.color,
		b->symmetry.x1, b->symmetry.x2, b->symmetry.y1, b->symmetry.y2, b->symmetry.d,
		b->symmetry.type,
	};
	memcpy(ints, scalars, sizeof(scalars));
	for (int i = 0; i < S_MAX; i++)
		captures[i] = b->captures[i];
	*komi = b->komi;
	for (int i = 0; i < 8; i++)
		nei8[i] = b->nei8[i];
	for (int i = 0; i < 4; i++)
		dnei[i] = b->dnei[i];
	for (int i = 0; i < b->size2; i++) {
		stones[i] = b->b[i];
		g[i] = b->g[i];
		p[i] = b->p[i];
		for (int s = 0; s < S_MAX; s++)
			n[i * S_MAX + s] = b->n[i].colors[s];
		h[2 * i] = b->h[2 * i];
		h[2 * i + 1] = b->h[2 * i + 1];
		pat3[i] = b->pat3[i];
		coord[2 * i] = b->coord[i][0];
		coord[2 * i + 1] = b->coord[i][1];
		for (int l = 0; l < GROUP_KEEP_LIBS; l++)
			gi_lib[i * GROUP_KEEP_LIBS + l] = b->gi[i].lib[l];
		gi_libs[i] = b->gi[i].libs;
	}
	for (int i = 0; i < b->flen; i++)
		f[i] = b->f[i];
	for (int i = 0; i < b->clen; i++)
		c[i] = b->c[i];
	for (int i = 0; i < 1 << history_hash_bits; i++)
		history_hash[i] = b->history_hash[i];
	*hash = b->hash;
	for (int i = 0; i < 4; i++)
		qhash[i] = b->qhash[i];
}

/* RULES_STONES_ONLY has no name in board_set_rules(). */
void ref_board_set_rules_id(void *b, int rules) { ((struct board *)b)->rules = rules; }

/* ---- Patterns and tactics for the tests of pattern3.py and tactics/. */

#include "pattern3.h"
#include "tactics/1lib.h"
#include "tactics/2lib.h"
#include "tactics/ladder.h"
#include "tactics/nakade.h"
#include "tactics/nlib.h"
#include "tactics/selfatari.h"
#include "tactics/util.h"

unsigned long ref_hash3_to_hash(unsigned int pat) { return hash3_to_hash(pat); }
unsigned int ref_pattern3_reverse(unsigned int pat) { return pattern3_reverse(pat); }

void
ref_pattern3_transpose(unsigned int pat, unsigned int *out)
{
	hash3_t transp[8];
	pattern3_transpose(pat, &transp);
	for (int i = 0; i < 8; i++)
		out[i] = transp[i];
}

/* @src: n patterns of 10 characters, one after another. */
void *
ref_pattern3s_new(const char *src, int n)
{
	char (*lines)[11] = calloc(n, 11);
	for (int i = 0; i < n; i++)
		memcpy(lines[i], src + 10 * i, 10);
	struct pattern3s *p = calloc(1, sizeof(*p));
	pattern3s_init(p, lines, n);
	free(lines);
	return p;
}

void
ref_pattern3s_dump(void *handle, unsigned int *pattern, unsigned char *value)
{
	struct pattern3s *p = handle;
	for (int i = 0; i < pattern3_hash_size; i++) {
		pattern[i] = p->hash[i].pattern;
		value[i] = p->hash[i].value;
	}
}

int
ref_pattern3_move_here(void *p, void *b, int coord, int color)
{
	struct move m = { coord, color };
	char idx = -1;
	if (!pattern3_move_here(p, b, &m, &idx))
		return -1;
	return idx;
}

static void
queue_out(struct move_queue *q, int *moves, int *tags, int *n)
{
	for (unsigned int i = 0; i < q->moves; i++) {
		moves[i] = q->move[i];
		tags[i] = q->tag[i];
	}
	*n = q->moves;
}

int ref_is_bad_selfatari(void *b, int color, int to) { return is_bad_selfatari(b, color, to); }

int
ref_selfatari_cousin(void *b, int color, int coord, int *bygroup)
{
	group_t g = 0;
	coord_t c = selfatari_cousin(b, color, coord, &g);
	*bygroup = g;
	return c;
}

int
ref_can_countercapture(void *b, int owner, int g, int to_play, int use_queue, int tag,
		       int *moves, int *tags, int *n)
{
	struct move_queue q = { .moves = 0 };
	int can = can_countercapture(b, owner, g, to_play, use_queue ? &q : NULL, tag);
	queue_out(&q, moves, tags, n);
	return can;
}

/* Returns the ladder move (or -3 if not set). */
int
ref_group_atari_check(unsigned int alwaysccaprate, void *b, int group, int to_play,
		      int use_ladder, int middle_ladder, int tag, int *moves, int *tags, int *n)
{
	struct move_queue q = { .moves = 0 };
	coord_t ladder = -3;
	group_atari_check(alwaysccaprate, b, group, to_play, &q, use_ladder ? &ladder : NULL,
			  middle_ladder, tag);
	queue_out(&q, moves, tags, n);
	return ladder;
}

void
ref_can_atari_group(void *b, int group, int owner, int to_play, int tag, int use_def_no_hopeless,
		    int *moves, int *tags, int *n)
{
	struct move_queue q = { .moves = 0 };
	can_atari_group(b, group, owner, to_play, &q, tag, use_def_no_hopeless);
	queue_out(&q, moves, tags, n);
}

void
ref_group_2lib_check(void *b, int group, int to_play, int tag, int use_miaisafe,
		     int use_def_no_hopeless, int *moves, int *tags, int *n)
{
	struct move_queue q = { .moves = 0 };
	group_2lib_check(b, group, to_play, &q, tag, use_miaisafe, use_def_no_hopeless);
	queue_out(&q, moves, tags, n);
}

void
ref_group_nlib_defense_check(void *b, int group, int to_play, int tag, int *moves, int *tags, int *n)
{
	struct move_queue q = { .moves = 0 };
	group_nlib_defense_check(b, group, to_play, &q, tag);
	queue_out(&q, moves, tags, n);
}

int ref_is_ladder(void *b, int coord, int laddered, int test_middle) { return is_ladder(b, coord, laddered, test_middle); }

int
ref_wouldbe_ladder(void *b, int group, int escapelib, int chaselib, int lcolor)
{
	return wouldbe_ladder(b, group, escapelib, chaselib, lcolor);
}

int ref_nakade_point(void *b, int around, int color) { return nakade_point(b, around, color); }
int ref_nakade_dead_shape(void *b, int around, int color) { return nakade_dead_shape(b, around, color); }
int ref_board_stone_radar(void *b, int coord, int distance) { return board_stone_radar(b, coord, distance); }
void ref_cfg_distances(void *b, int start, int *distances, int maxdist) { cfg_distances(b, start, distances, maxdist); }
int ref_coord_edge_distance(void *b, int c) { return coord_edge_distance(c, b); }
int ref_coord_gridcular_distance(void *b, int c1, int c2) { return coord_gridcular_distance(c1, c2, b); }
float ref_board_effective_handicap(void *b, int value) { return board_effective_handicap(b, value); }
int ref_board_estimated_moves_left(void *b) { return board_estimated_moves_left(b); }
double ref_board_local_value(int scan_neis, void *b, int coord, int color) { return board_local_value(scan_neis, b, coord, color); }

/* ---- Playouts for the tests of playout/. */

#include "ownermap.h"
#include "playout.h"
#include "playout/light.h"
#include "playout/moggy.h"
#include "uct/prior.h"

void *
ref_moggy_new(const char *arg, void *b)
{
	char *copy = arg ? strdup(arg) : NULL;
	struct playout_policy *p = playout_moggy_init(copy, b, NULL);
	return p;
}

void *
ref_light_new(void *b)
{
	return playout_light_init(NULL, b);
}

int
ref_policy_choose(void *policy, void *b, int color)
{
	struct playout_policy *p = policy;
	struct playout_setup setup = { .gamelen = MAX_GAMELEN };
	return p->choose(p, &setup, b, color);
}

int
ref_moggy_permit(void *policy, void *b, int coord, int color, int *out_coord)
{
	struct playout_policy *p = policy;
	struct move m = { coord, color };
	int permitted = p->permit(p, b, &m);
	*out_coord = m.coord;
	return permitted;
}

int
ref_play_random_move(void *policy, void *b, int color)
{
	struct playout_setup setup = { .gamelen = MAX_GAMELEN };
	return play_random_move(&setup, b, color, policy);
}

int
ref_play_random_game(void *policy, void *b, int color, int gamelen, int mercymin,
		     int *amaf_game, unsigned char *amaf_ko, int *amaf_len,
		     int *owner_map, int *owner_playouts)
{
	struct board *board = b;
	struct playout_setup setup = { .gamelen = gamelen, .mercymin = mercymin };
	struct playout_amafmap amaf;
	amaf.gamelen = amaf.game_baselen = 0;
	struct board_ownermap ownermap;
	ownermap.playouts = 0;
	ownermap.map = calloc(board_size2(board), sizeof(ownermap.map[0]));
	int result = play_random_game(&setup, board, color, &amaf, &ownermap, policy);
	for (int i = 0; i < amaf.gamelen; i++) {
		amaf_game[i] = amaf.game[i];
		amaf_ko[i] = amaf.is_ko_capture[i];
	}
	*amaf_len = amaf.gamelen;
	for (int i = 0; i < board_size2(board); i++)
		for (int s = 0; s < S_MAX; s++)
			owner_map[i * S_MAX + s] = ownermap.map[i][s];
	*owner_playouts = ownermap.playouts;
	free(ownermap.map);
	return result;
}

void
ref_policy_assess(void *policy, void *board, int to_play, int parity, const unsigned char *consider,
		  int games, float *prior_value, int *prior_playouts)
{
	struct playout_policy *p = policy;
	struct board *b = board;
	int size2 = board_size2(b);
	struct move_stats *priors = calloc(size2, sizeof(*priors));
	bool *considers = calloc(size2, sizeof(*considers));
	for (int i = 0; i < size2; i++)
		considers[i] = consider[i];
	struct prior_map map = { .b = b, .to_play = to_play, .parity = parity,
				 .prior = priors, .consider = considers };
	if (p->assess)
		p->assess(p, &map, games);
	for (int i = 0; i < size2; i++) {
		prior_value[i] = priors[i].value;
		prior_playouts[i] = priors[i].playouts;
	}
	free(priors);
	free(considers);
}
