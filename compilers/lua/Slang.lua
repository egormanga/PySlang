-- Slang stdlib for Lua --


_meta = {
	__call = function(self, obj) obj = type(obj) == 'table' and obj or {obj}; self.__index = self; return setmetatable(obj, self) end,
}

bool = setmetatable({
	__tostring = function(x) return tostring(x[1]) end,
	__bool = function(x) return x[1] end,
	__in = function(a, b) return bool(b:__has(a[1])) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(bool == b) end,
	__not = function(x) return bool(not x:__bool()) end,
	__and = function(a, b) return bool(a[1] and b) end,
	__or = function(a, b) return bool(a[1] or b) end,
	__xor = function(a, b) return bool((a[1] and 1 or 0) ~ (b and 1 or 0) and (a[1] or b)) end,
	__eq = function(a, b) return (a[1] == b) end,
	__band = function(a, b) return bool((getmetatable(a) == bool and a[1] or a) & (getmetatable(b) == bool and b[1] or b)) end,
	__bor = function(a, b) return bool((getmetatable(a) == bool and a[1] or a) | (getmetatable(b) == bool and b[1] or b)) end,
	__bxor = function(a, b) return bool((getmetatable(a) == bool and a[1] or a) ~ (getmetatable(b) == bool and b[1] or b)) end,
	__bnot = function(x) return bool(~x[1]) end,
}, _meta)

int = setmetatable({
	__tostring = function(x) return tostring(x[1]) end,
	__bool = function(x) return (x[1] ~= 0) end,
	__unp = function(x) return int(math.abs(x[1])) end,
	__unm = function(x) return int(-x[1]) end,
	__preinc = function(x) x[1] = (x[1] + 1); return int(x[1]) end,
	__predec = function(x) x[1] = (x[1] - 1); return int(x[1]) end,
	__postinc = function(x) local r<const> = x[1]; x[1] = (x[1] + 1); return int(r) end,
	__postdec = function(x) local r<const> = x[1]; x[1] = (x[1] - 1); return int(r) end,
	__sqr = function(x) return int(x[1] * x[1]) end,
	__in = function(a, b) return bool(b:__has(a[1])) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(int == b) end,
	__not = function(x) return bool(not x:__bool()) end,
	__and = function(a, b) return bool(a[1] and b) end,
	__or = function(a, b) return bool(a[1] or b) end,
	__xor = function(a, b) return bool((a[1] and 1 or 0) ~ (b and 1 or 0) and (a[1] or b)) end,
	__to = function(a, b) return range{a[1], b[1]} end,
	__eq = function(a, b) return ((getmetatable(a) == int and a[1] or a) == (getmetatable(b) == int and b[1] or b)) end,
	__lt = function(a, b) return ((getmetatable(a) == int and a[1] or a) < (getmetatable(b) == int and b[1] or b)) end,
	__le = function(a, b) return ((getmetatable(a) == int and a[1] or a) <= (getmetatable(b) == int and b[1] or b)) end,
	__add = function(a, b) return int((getmetatable(a) == int and a[1] or a) + (getmetatable(b) == int and b[1] or b)) end,
	__sub = function(a, b) return int((getmetatable(a) == int and a[1] or a) - (getmetatable(b) == int and b[1] or b)) end,
	__mul = function(a, b) return int((getmetatable(a) == int and a[1] or a) * (getmetatable(b) == int and b[1] or b)) end,
	__div = function(a, b) return ((getmetatable(a) == int and float(a[1]) or a) / (getmetatable(b) == int and float(b[1]) or b)) end,
	__idiv = function(a, b) return int((getmetatable(a) == int and a[1] or a) // (getmetatable(b) == int and b[1] or b)) end,
	__mod = function(a, b) return int((getmetatable(a) == int and a[1] or a) % (getmetatable(b) == int and b[1] or b)) end,
	__pow = function(a, b) return int((getmetatable(a) == int and a[1] or a) ^ (getmetatable(b) == int and b[1] or b)) end,
	__shl = function(a, b) return int((getmetatable(a) == int and a[1] or a) << (getmetatable(b) == int and b[1] or b)) end,
	__shr = function(a, b) return int((getmetatable(a) == int and a[1] or a) >> (getmetatable(b) == int and b[1] or b)) end,
	__band = function(a, b) return int((getmetatable(a) == int and a[1] or a) & (getmetatable(b) == int and b[1] or b)) end,
	__bor = function(a, b) return int((getmetatable(a) == int and a[1] or a) | (getmetatable(b) == int and b[1] or b)) end,
	__bxor = function(a, b) return int((getmetatable(a) == int and a[1] or a) ~ (getmetatable(b) == int and b[1] or b)) end,
	__bnot = function(x) return int(~x[1]) end,
	popcount = function(self)
		r = ((self[1] < 0) and 1 or 0)
		x = math.abs(self[1])
		while x ~= 0 do
			r = (r + (x & 1))
			x = (x >> 1)
		end
		return int(r)
	end,
	tobase = function(self, base)
		if base < 1 or base > 36 then error"Invalid base" end
		x = self[1]
		if x == 0 then return str'0' end
		if base == 10 then return str(tostring(x)) end
		if base == 1 then return str(('1'):rep(x)) end
		local digits<const> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		local sign = ''
		if x < 0 then sign = '-'; x = -x end
		local r = {}
		repeat
			local d<const> = (1 + (x % base))
			x = (x // base)
			table.insert(r, 1, digits:sub(d, d))
		until x == 0
		return str(sign .. table.concat(r, ''))
	end,
	length = function(self, base) base = base or 2
		x = self[1]
		if x == 0 then return int(1) end
		if base == 10 then return int(#tostring(x)) end
		if base == 1 then return int(x) end
		local r = 0
		if x < 0 then r = 1; x = -x end
		repeat
			r = (r + 1)
			x = (x // base)
		until x == 0
		return int(r)
	end,
}, _meta)

float = setmetatable({
	__tostring = function(x) return tostring(x[1]) end,
	__bool = function(x) return (x[1] ~= 0) end,
	__unp = function(x) return float(math.abs(x[1])) end,
	__unm = function(x) return float(-x[1]) end,
	__in = function(a, b) return bool(b:__has(a[1])) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(float == b) end,
	__not = function(x) return bool(not x:__bool()) end,
	__and = function(a, b) return bool(a[1] and b) end,
	__or = function(a, b) return bool(a[1] or b) end,
	__xor = function(a, b) return bool((a[1] and 1 or 0) ~ (b and 1 or 0) and (a[1] or b)) end,
	__eq = function(a, b) return ((getmetatable(a) == float and a[1] or a) == (getmetatable(b) == float and b[1] or b)) end,
	__lt = function(a, b) return ((getmetatable(a) == float and a[1] or a) < (getmetatable(b) == float and b[1] or b)) end,
	__le = function(a, b) return ((getmetatable(a) == float and a[1] or a) <= (getmetatable(b) == float and b[1] or b)) end,
	__add = function(a, b) return float((getmetatable(a) == float and a[1] or a) + (getmetatable(b) == float and b[1] or b)) end,
	__sub = function(a, b) return float((getmetatable(a) == float and a[1] or a) - (getmetatable(b) == float and b[1] or b)) end,
	__mul = function(a, b) return float((getmetatable(a) == float and a[1] or a) * (getmetatable(b) == float and b[1] or b)) end,
	__div = function(a, b) return float((getmetatable(a) == float and a[1] or a) / (getmetatable(b) == float and b[1] or b)) end,
	__idiv = function(a, b) return int((getmetatable(a) == float and a[1] or a) // (getmetatable(b) == float and b[1] or b)) end,
	__pow = function(a, b) return float((getmetatable(a) == float and a[1] or a) ^ (getmetatable(b) == float and b[1] or b)) end,
	round = function(self) return int(self[1] + (2^52 + 2^51) - (2^52 + 2^51)) end,
	iswhole = function(self) return bool(self[1] % 1 == 0) end,
}, _meta)

char = setmetatable({
	__tostring = function(x) return ("'" .. x[1]:gsub("'", "\'") .. "'") end,
	__bool = function(x) return (x[1] ~= '') end,
	__in = function(a, b) return bool(b:__has(a[1])) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(char == b) end,
	__not = function(x) return bool(not x:__bool()) end,
	__and = function(a, b) return bool(a[1] and b) end,
	__or = function(a, b) return bool(a[1] or b) end,
	__xor = function(a, b) return bool((a[1] and 1 or 0) ~ (b and 1 or 0) and (a[1] or b)) end,
	__to = function(a, b) return range{a[1], b[1]} end,
	__eq = function(a, b) return ((getmetatable(a) == char and a[1] or a) == (getmetatable(b) == char and b[1] or b)) end,
	__add = function(a, b) return str((getmetatable(a) == char and a[1] or a) .. (getmetatable(b) == char and b[1] or b)) end,
	__mul = function(a, b) return str(string.rep((getmetatable(a) == char and a[1] or b[1]), (getmetatable(b) ~= str and b[1] or a[1]))) end,
	__concat = function(a, b) return str(a + b) end,
	__length = function(x) return int(string.len(x[1])) end,
}, _meta)

str = setmetatable({
	__tostring = function(x) return x[1]:format('%q'):gsub('\n', '\\n') end,
	__bool = function(x) return (x[1] ~= '') end,
	__in = function(a, b) return bool(b:__has(a[1])) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(str == b) end,
	__not = function(x) return bool(not x:__bool()) end,
	__and = function(a, b) return bool(a[1] and b) end,
	__or = function(a, b) return bool(a[1] or b) end,
	__xor = function(a, b) return bool((a[1] and 1 or 0) ~ (b and 1 or 0) and (a[1] or b)) end,
	__eq = function(a, b) return ((getmetatable(a) == str and a[1] or a) == (getmetatable(b) == str and b[1] or b)) end,
	__add = function(a, b) return str((getmetatable(a) == str and a[1] or a) .. (getmetatable(b) == str and b[1] or b)) end,
	__mul = function(a, b) return str(string.rep((getmetatable(a) == str and a[1] or b[1]), (getmetatable(b) ~= str and b[1] or a[1]))) end,
	__concat = function(a, b) return str(a + b) end,
	__length = function(x) return int(string.len(x[1])) end,
	__index = function(a, b) return char(a[1][b]) end,
	length = function(self) return #self[1] end,
	strip = function(self) return str(self[1]:gsub('^%s+|%s+$', '')) end,
}, _meta)

range = setmetatable({
	__pairs = function(x)
		local from<const>, to<const> = table.unpack(x)
		return function (_, i)
			i = (i + 1)
			if i >= to then return nil end
			return i
		end, nil, from-1
	end,
}, _meta)

list = setmetatable({
	__tostring = function(x)
		r = '['
		for ii, i in ipairs(self) do
			if ii > 1 then r = (r + ', ') end
			r = (r + tostring(i))
		end
		return (r .. ']')
	end,
	__bool = function(x) return (#x ~= 0) end,
	__has = function(a, b)
		for ii, i in ipairs(a) do
			if i == b then return true end
		end
		return false
	end,
	__in = function(a, b) return bool(b:__has(a)) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(list == b) end,
	__not = function(x) return bool(not x:__bool()) end,
	__and = function(a, b) return bool(#a and #b) end,
	__or = function(a, b) return bool(#a or #b) end,
	__xor = function(a, b) return bool((#a and 1 or 0) ~ (#b and 1 or 0) and (a or b)) end,
	__eq = function(a, b) return (a == b) end,
	append = function(self, item) table.insert(self, item) end,
	insert = function(self, index, item) table.insert(self, index, item) end,
	pop = function(self, index) return table.remove(self, index) end,
	reverse = function(self)
		l = #self
		for i = 1, l//2 do
			self[i], self[l-i+1] = self[l-i+1], self[i]
		end
	end,
	sort = function(self) table.sort(self) end,
	each = function(x, function_)
		for ii, i in ipairs(x) do
			function_(i)
		end
	end,
}, _meta)

_function = setmetatable({
	__tostring = function(x) return tostring(x[1]) end,
	__in = function(a, b) return bool(b:__has(a[1])) end,
	__not_in = function(a, b) return ~a:__in(b) end,
	__isof = function(a, b) return bool(_function == b) end,
	__and = function(a, b) return bool(a[1] and b) end,
	__or = function(a, b) return bool(a[1] or b) end,
	__xor = function(a, b) return bool((a[1] and 1 or 0) ~ (b and 1 or 0) and (a[1] or b)) end,
	__eq = function(a, b) return (a[1] == b) end,
	__call = function(x, ...) return x[1](...) end,
	map = function(x, function_)
		r = {}
		for ii, i in pairs(x) do
			table.insert(r, function_(i))
		end
		return table.unpack(r)
	end,
}, _meta)

_meta = nil


stdio = setmetatable({
	readln = function(self) io.read() end,
	print = function(self, ...)
		for ii, i in ipairs(table.pack(...)) do
			io.write((ii > 1 and ' ' or ''), tostring(i))
		end
	end,
	println = function(self, ...) stdio:print(...); io.write('\n') end,
}, {__name = 'stdio'})


--[[ by Sdore, 2024
     slang.sdore.me ]]--
