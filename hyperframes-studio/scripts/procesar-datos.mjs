
import fs from "node:fs";
import path from "node:path";

const entrada = "data/input/datos.csv";
const salida = "data/generated/datos.js";

function procesarDatos() {

    if (!fs.existsSync(entrada)) {
        throw new Error("No se encontró el archivo CSV");
    }

    const contenido = fs.readFileSync(entrada, "utf8");

    const filas = contenido.trim().split(/\r?\n/);

    const encabezados = filas[0].split(",").map(v => v.trim());

    if (!encabezados.includes("x") ||
        !encabezados.includes("y")) {
        throw new Error("Se necesitan las columnas x e y");
    }

    const ix = encabezados.indexOf("x");
    const iy = encabezados.indexOf("y");

    const datos = filas.slice(1).map((fila, i) => {

        const columnas = fila.split(",");

        const x = Number(columnas[ix]);
        const y = Number(columnas[iy]);

        if (!Number.isFinite(x) || !Number.isFinite(y)) {
            throw new Error(`Dato inválido en la fila ${i + 2}`);
        }

        return { x, y };
    });

    if (datos.length < 2) {
        throw new Error("Se necesitan al menos dos registros");
    }

    const n = datos.length;

    const sx = datos.reduce((s, p) => s + p.x, 0);
    const sy = datos.reduce((s, p) => s + p.y, 0);

    const sxx = datos.reduce((s, p) => s + p.x ** 2, 0);
    const sxy = datos.reduce((s, p) => s + p.x * p.y, 0);

    const denominador = n * sxx - sx ** 2;

    if (Math.abs(denominador) < 1e-12) {
        throw new Error("No existe una pendiente única");
    }

    const a = (n * sxy - sx * sy) / denominador;
    const b = (sy - a * sx) / n;

    const error = datos.reduce((s, p) => {
        return s + (p.y - (a * p.x + b)) ** 2;
    }, 0);

    const resultado = {
        total: n,
        pendiente: a,
        intercepto: b,
        errorCuadratico: error,
        ecuacion: `y = ${a.toFixed(3)}x + ${b.toFixed(3)}`,
        datos
    };

    fs.mkdirSync(path.dirname(salida), {
        recursive: true
    });

    fs.writeFileSync(
        salida,
        `window.HYPEFRAMES_DATA = ${JSON.stringify(resultado, null, 2)};`
    );

    console.log("Datos procesados correctamente");
    console.log(resultado);
}

try {
    procesarDatos();
} catch (error) {
    console.error("ERROR:", error.message);
    process.exitCode = 1;
}
