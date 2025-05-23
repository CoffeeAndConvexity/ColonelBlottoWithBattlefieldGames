IDIR = include
LDIR = /opt/homebrew/lib
CC=clang++
CFLAGS=-Ofast -I$(IDIR) -I$(SCHEMA_DIR) -L/opt/homebrew/opt/llvm/lib -std=c++17 #-Xpreprocessor -fopenmp -lomp
ODIR=obj

_DEPS = range.hpp util.h id_types.h logger.h enumerate.hpp regret_minimizer.h regret_minimizer_efce_flattened_base.h rmplus_efce_flattened.h rm_efce_flattened.h prm_efce_flattened.h prmplus_efce_flattened.h battlefield_level_game.h blotto.h blotto_alt.h
DEPS = $(patsubst %, $(IDIR)/%, $(_DEPS))

# UNCOMMENT FOR NO ENHANCEMENT (and vice versa)
_OBJ = blotto_basic.o blotto.o regret_minimizer_efce_flattened_base.o rmplus_efce_flattened.o prm_efce_flattened.o rm_efce_flattened.o leaf.o prmplus_efce_flattened.o battlefield_level_game.o 

# UNCOMMENT FOR ENHANCEMENT (and vice versa)
# _OBJ = blotto_basic_speedup.o blotto_alt.o regret_minimizer_efce_flattened_base.o rmplus_efce_flattened.o prm_efce_flattened.o rm_efce_flattened.o leaf.o prmplus_efce_flattened.o battlefield_level_game.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


#===============================
# UNCOMMENT FOR NO ENHANCEMENT (and vice versa)
blotto_basic: $(OBJ) obj/blotto_basic.o 
	$(CC) -o $@ $^ $(LIBS) $(CFLAGS) -v

obj/blotto_basic.o: src/blotto_basic.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

obj/blotto.o: src/blotto.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

#===================================

#======================================
# UNCOMMENT FOR ENHANCEMENT (and vice versa)
# blotto_alt: $(OBJ) obj/blotto_basic_speedup.o 
#	$(CC) -o $@ $^ $(LIBS) $(CFLAGS) -v

#obj/blotto_basic_speedup.o: src/blotto_basic_speedup.cpp $(DEPS)
#	$(CC) -c -o $@ $< $(CFLAGS)

#obj/blotto_alt.o: src/blotto_alt.cpp $(DEPS)
#	$(CC) -c -o $@ $< $(CFLAGS)
#=======================================

obj/regret_minimizer_efce_flattened_base.o: src/regret_minimizer_efce_flattened_base.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/rmplus_efce_flattened.o: src/rmplus_efce_flattened.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/rm_efce_flattened.o: src/rm_efce_flattened.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/prm_efce_flattened.o: src/prm_efce_flattened.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/prmplus_efce_flattened.o: src/prmplus_efce_flattened.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/rm_unrolled_scaled_extension.o: src/rm_unrolled_scaled_extension.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/leaf.o: src/leaf.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
obj/battlefield_level_game.o: src/battlefield_level_game.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


.PHONY: clean

clean: 
	rm -f $(ODIR)/*.o 